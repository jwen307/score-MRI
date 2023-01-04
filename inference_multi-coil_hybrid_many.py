from pathlib import Path

import network_utils
import fastmri
import viz
from models import utils as mutils
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector,
                      get_pc_fouriercs_RI_coil_SENSE)
from models import ncsnpp
import time
from utils import fft2_m, ifft2_m, get_mask, get_data_scaler, get_data_inverse_scaler, restore_checkpoint, \
    normalize_complex, root_sum_of_squares, lambda_schedule_const, lambda_schedule_linear
import torch
import torch.nn as nn
import numpy as np
from models.ema import ExponentialMovingAverage
import matplotlib.pyplot as plt
import importlib
import argparse
import sigpy.mri as mr
import sigpy as sp
import torchvision
from fastmri.data.transforms import to_tensor, tensor_to_complex_np

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from typing import Optional



def main():
    ###############################################
    # 1. Configurations
    ###############################################

    # args
    args = create_argparser().parse_args()
    N = args.N
    m = args.m
    fname = args.data
    filename = f'./samples/multi-coil/{args.task}/{fname}.npy'
    mask_filename = f'./samples/multi-coil/prospective/{fname}_mask.npy'

    print('initializing...')
    configs = importlib.import_module(f"configs.ve.fastmri_knee_320_ncsnpp_continuous")
    config = configs.get_config()
    img_size = config.data.image_size
    batch_size = 1

    schedule = 'linear'
    start_lamb = 1.0
    end_lamb = 0.2
    m_steps = 50

    num_coils = 8

    if schedule == 'const':
        lamb_schedule = lambda_schedule_const(lamb=start_lamb)
    elif schedule == 'linear':
        lamb_schedule = lambda_schedule_linear(start_lamb=start_lamb, end_lamb=end_lamb)
    else:
        NotImplementedError(f"Given schedule {schedule} not implemented yet!")

    # Read data
    img = normalize_complex(torch.from_numpy(np.load(filename).astype(np.complex64)))
    # TODO: Changed to 8 virtual coils (JW)
    img = img.view(1, num_coils, 320, 320)
    img = img.to(config.device)

    # TODO: Changed to look for retrospective and prospective (JW)
    if args.task == 'retrospective':
        # generate mask
        mask = get_mask(img, img_size, batch_size,
                        type=args.mask_type,
                        acc_factor=args.acc_factor,
                        center_fraction=args.center_fraction)
    elif args.task == 'prospective':
        mask = torch.from_numpy(np.load(mask_filename))
        mask = mask.view(1, 1, 320, 320)

    ckpt_filename = f"./weights/checkpoint_95.pth"
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=N)

    config.training.batch_size = batch_size
    predictor = ReverseDiffusionPredictor
    corrector = LangevinCorrector
    probability_flow = False
    snr = 0.16

    # sigmas = mutils.get_sigmas(config)
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # create model and load checkpoint
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(),
                                   decay=config.model.ema_rate)
    state = dict(step=0, model=score_model, ema=ema)
    state = restore_checkpoint(ckpt_filename, state, config.device, skip_sigma=True)
    ema.copy_to(score_model.parameters())

    # Specify save directory for saving generated samples
    save_root = Path(f'./results/multi-coil/hybrid')
    save_root.mkdir(parents=True, exist_ok=True)

    irl_types = ['input', 'recon', 'recon_progress', 'label']
    for t in irl_types:
        save_root_f = save_root / t
        save_root_f.mkdir(parents=True, exist_ok=True)

    ###############################################
    # 2. Inference
    ###############################################
    mps_dir = save_root / f'sens.npy'
    # fft
    kspace = fft2_m(img)

    # undersampling
    under_kspace = kspace * mask
    under_img = ifft2_m(under_kspace)

    # ESPiRiT
    if mps_dir.exists():
        mps = np.load(str(mps_dir))
    else:
        mps = mr.app.EspiritCalib(kspace.cpu().detach().squeeze().numpy()).run()
        np.save(str(save_root / f'sens.npy'), mps)
    mps = torch.from_numpy(mps).view(1, num_coils, 320, 320).to(kspace.device)

    # Defines the reconstruction procedure
    pc_fouriercs = get_pc_fouriercs_RI_coil_SENSE(sde,
                                                  predictor, corrector,
                                                  inverse_scaler,
                                                  snr=snr,
                                                  n_steps=m,
                                                  m_steps=50,
                                                  mask=mask,
                                                  sens=mps,
                                                  lamb_schedule=lamb_schedule,
                                                  probability_flow=probability_flow,
                                                  continuous=config.training.continuous,
                                                  denoise=True)

    print(f'Beginning inference')
    tic = time.time()
    x = pc_fouriercs(score_model, scaler(under_img), y=under_kspace)
    toc = time.time() - tic
    print(f'Time took for recon: {toc} secs.')

    ###############################################
    # 3. Saving recon
    ###############################################
    under_img = root_sum_of_squares(under_img, dim=1)
    label = root_sum_of_squares(img, dim=1)
    input = under_img.squeeze().cpu().detach().numpy()
    label = label.squeeze().cpu().detach().numpy()
    mask_sv = mask[0, 0, :, :].squeeze().cpu().detach().numpy()

    np.save(str(save_root / 'input' / fname) + '.npy', input)
    np.save(str(save_root / 'input' / (fname + '_mask')) + '.npy', mask_sv)
    np.save(str(save_root / 'label' / fname) + '.npy', label)
    plt.imsave(str(save_root / 'input' / fname) + '.png', np.abs(input), cmap='gray')
    plt.imsave(str(save_root / 'label' / fname) + '.png', np.abs(label), cmap='gray')

    x = root_sum_of_squares(x, dim=1)
    recon = x.squeeze().cpu().detach().numpy()
    np.save(str(save_root / 'recon' / fname) + '.npy', recon)
    plt.imsave(str(save_root / 'recon' / fname) + '.png', np.abs(recon), cmap='gray')



def psnr(gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)

def ssim(
        gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    # if not gt.ndim == 3:
    #    raise ValueError("Unexpected number of dimensions in ground truth.")
    if gt.shape[-1] == 2:
        is_complex = True
    else:
        is_complex = False

    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    ssim = 0
    for slice_num in range(gt.shape[0]):
        if is_complex:
            ssim = ssim + structural_similarity(
                gt[slice_num], pred[slice_num], channel_axis=-1, data_range=maxval
            )
        else:
            ssim = ssim + structural_similarity(
                gt[slice_num], pred[slice_num], data_range=maxval
            )

    return ssim / gt.shape[0]


# %%
if __name__ == "__main__":
    ###############################################
    # 1. Configurations
    ###############################################

    # args
    N = 2000
    m = 1
    img_num = 100
    fname = 'val{}'.format(img_num)
    task = 'prospective'
    filename = f'./samples/multi-coil/prospective/{fname}.npy'
    mask_filename = f'./samples/multi-coil/prospective/{fname}_mask.npy'

    print('initaializing...')
    configs = importlib.import_module(f"configs.ve.fastmri_knee_320_ncsnpp_continuous")
    config = configs.get_config()
    img_size = config.data.image_size
    batch_size = 1

    schedule = 'linear'
    start_lamb = 1.0
    end_lamb = 0.2
    m_steps = 50

    num_coils = 8
    num_acs = 13

    if schedule == 'const':
        lamb_schedule = lambda_schedule_const(lamb=start_lamb)
    elif schedule == 'linear':
        lamb_schedule = lambda_schedule_linear(start_lamb=start_lamb, end_lamb=end_lamb)
    else:
        NotImplementedError(f"Given schedule {schedule} not implemented yet!")

    # Read data
    img = normalize_complex(torch.from_numpy(np.load(filename).astype(np.complex64)))
    # TODO: Changed to 8 virtual coils (JW)
    img = img.view(1, num_coils, 320, 320)
    img = img.to(config.device)

    # TODO: Changed to look for retrospective and prospective (JW)
    if task == 'retrospective':
        # generate mask
        mask = get_mask(img, img_size, batch_size,
                        type=args.mask_type,
                        acc_factor=args.acc_factor,
                        center_fraction=args.center_fraction)
    elif task == 'prospective':
        mask = torch.from_numpy(np.load(mask_filename))
        mask = mask.view(1, 1, 320, 320).to(config.device)

    ckpt_filename = f"./weights/checkpoint_95.pth"
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=N)

    config.training.batch_size = batch_size
    predictor = ReverseDiffusionPredictor
    corrector = LangevinCorrector
    probability_flow = False
    snr = 0.16

    # sigmas = mutils.get_sigmas(config)
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # create model and load checkpoint
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(),
                                   decay=config.model.ema_rate)
    state = dict(step=0, model=score_model, ema=ema)
    state = restore_checkpoint(ckpt_filename, state, config.device, skip_sigma=True)
    ema.copy_to(score_model.parameters())

    # Specify save directory for saving generated samples
    save_root = Path(f'./results/multi-coil/hybrid')
    save_root.mkdir(parents=True, exist_ok=True)

    irl_types = ['input', 'recon', 'recon_progress', 'label']
    for t in irl_types:
        save_root_f = save_root / t
        save_root_f.mkdir(parents=True, exist_ok=True)
#%%
    ###############################################
    # 2. Inference
    ###############################################
    mps_dir = save_root / f'sens.npy'
    # fft
    kspace = fft2_m(img)

    # undersampling
    under_kspace = kspace * mask
    under_img = ifft2_m(under_kspace)

    # ESPiRiT
    # if mps_dir.exists():
    #     mps = np.load(str(mps_dir))
    # else:
    mps = mr.app.EspiritCalib(under_kspace.cpu().detach().squeeze().numpy(),
                              calib_width=num_acs,
                              crop=0.7,
                              #device=sp.Device(0),
                              kernel_width=6).run()
        #np.save(str(save_root / f'sens.npy'), mps)
    mps = torch.from_numpy(mps).view(1, num_coils, 320, 320).to(kspace.device)

    # Defines the reconstruction procedure
    pc_fouriercs = get_pc_fouriercs_RI_coil_SENSE(sde,
                                                  predictor, corrector,
                                                  inverse_scaler,
                                                  snr=snr,
                                                  n_steps=m,
                                                  m_steps=m_steps,
                                                  mask=mask,
                                                  sens=mps,
                                                  lamb_schedule=lamb_schedule,
                                                  probability_flow=probability_flow,
                                                  continuous=config.training.continuous,
                                                  denoise=True)

#%% Get multiple posterior samples
    print(f'Beginning inference')
    tic = time.time()

    #Get 32 posterior samples
    samples = []
    for i in range(8):
        #Get the samples
        x = pc_fouriercs(score_model, scaler(under_img), y=under_kspace)

        samples.append(x)

    toc = time.time() - tic
    print(f'Time took for recon: {toc} secs.')
    
#%% Unnormalize everything
img = utils.unnormalize_complex(img1, min_max)

unnorm_samples = []
for i in range(len(samples)):
    unnorm_samples.append(utils.unnormalize_complex(samples[i], min_max))


#%% Find the PSNR for different posterior samples
label = root_sum_of_squares(img, dim=1)
label = label.squeeze().cpu().detach().numpy()



num = [1,2,4,8] 
psnr_vals = []
ssim_vals = []
for n in num:
    
    recon = torch.stack(unnorm_samples[:n]).mean(dim=0).cpu()
    #recon = unnorm_samples[:n].mean(dim=0).cpu()
    
    recon = root_sum_of_squares(recon, dim=1)
    recon = recon.squeeze().cpu().detach().numpy()
    
    psnr_vals.append(psnr(np.abs(label), np.abs(recon)))
    ssim_vals.append(ssim(np.abs(label), np.abs(recon)))
    
#%% Plot the theoretical vs experiment PSNR vs num samples
vals = np.arange(1,num[-1],1)
theoretical = psnr_vals[-1] - (2.877 - 10*np.log10((2*vals)/(vals+1)))

plt.plot(vals,theoretical, label='Theoretical')
plt.plot(num,psnr_vals, label='Experimental')
plt.xlabel('Number of Samples')
plt.ylabel('PSNR')
plt.title('PSNR vs Number of Samples')
plt.legend()

#%% Plot the other way
vals = np.arange(1,num[-1],1)
theoretical = psnr_vals[0] + (10*np.log10((2*vals)/(vals+1)))

plt.plot(vals,theoretical, label='Theoretical')
plt.plot(num,psnr_vals, label='Experimental')
plt.xlabel('Number of Samples')
plt.ylabel('PSNR')
plt.title('PSNR vs Number of Samples')
plt.legend()



#%% View the mean image and standard deviation
recons = torch.stack(samples)
#recons = samples
mean_img = root_sum_of_squares(recons.mean(dim=0), dim=1).cpu()

viz.show_img(np.abs(mean_img))

imgs = np.abs(root_sum_of_squares(recons.squeeze(1),dim=1).cpu())

viz.show_std_map(imgs.unsqueeze(1))

#%% Save the recons
recon_np = torch.stack(samples).cpu().numpy()
np.save(str(save_root / 'recon' / fname) + 'many_{0}_N_{1}.npy'.format(img_num, N), recon_np)

#%%

#     ###############################################
#     # 3. Saving recon
#     ###############################################
#     under_img1 = root_sum_of_squares(under_img, dim=1)
#     label = root_sum_of_squares(img, dim=1)
#     input = under_img1.squeeze().cpu().detach().numpy()
#     label = label.squeeze().cpu().detach().numpy()
#     mask_sv = mask[0, 0, :, :].squeeze().cpu().detach().numpy()

#     np.save(str(save_root / 'input' / fname) + '.npy', input)
#     np.save(str(save_root / 'input' / (fname + '_mask')) + '.npy', mask_sv)
#     np.save(str(save_root / 'label' / fname) + '.npy', label)
#     plt.imsave(str(save_root / 'input' / fname) + '_rss.png', np.abs(input), cmap='gray')
#     plt.imsave(str(save_root / 'label' / fname) + '_rss.png', np.abs(label), cmap='gray')

#     x1 = root_sum_of_squares(x, dim=1)
#     recon = x1.squeeze().cpu().detach().numpy()
#     np.save(str(save_root / 'recon' / fname) + '.npy', recon)
#     plt.imsave(str(save_root / 'recon' / fname) + '_rss.png', np.abs(recon), cmap='gray')

#     psnr_val = psnr(np.abs(label), np.abs(recon))
#     print(psnr_val)


# #%% Do the reconstructions with predicted map
#     under_img2 = network_utils.multicoil2single(to_tensor(under_img.cpu()),mps)
#     gt_img2 = network_utils.multicoil2single(to_tensor(img.cpu()), mps)
#     recon_img2 = network_utils.multicoil2single(to_tensor(x.cpu()), mps)

#     plt.imsave(str(save_root / 'input' / fname) + '_maps.png', fastmri.complex_abs(under_img2).squeeze().numpy(), cmap='gray')
#     plt.imsave(str(save_root / 'label' / fname) + '_maps.png', fastmri.complex_abs(gt_img2).squeeze().numpy(), cmap='gray')
#     plt.imsave(str(save_root / 'recon' / fname) + '_maps.png', fastmri.complex_abs(recon_img2).squeeze().numpy(), cmap='gray')

# #%%
#     psnr_val_maps = psnr(fastmri.complex_abs(gt_img2).numpy(), fastmri.complex_abs(recon_img2).numpy())
#     print(psnr_val_maps)

