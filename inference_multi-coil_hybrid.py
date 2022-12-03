from pathlib import Path
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

    #Defines the reconstruction procedure
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


def create_argparser():
    parser = argparse.ArgumentParser()
    # TODO: Added task (JW)
    parser.add_argument('--task', choices=['retrospective', 'prospective'], default='retrospective',
                        type=str, help='If retrospective, under-samples the fully-sampled data with generated mask.'
                                       'If prospective, runs score-POCS with the given mask')
    parser.add_argument('--data', type=str, help='which data to use for reconstruction', required=True)
    parser.add_argument('--mask_type', type=str, help='which mask to use for retrospective undersampling.'
                                                      '(NOTE) only used for retrospective model!', default='gaussian1d',
                        choices=['gaussian1d', 'uniform1d', 'gaussian2d'])
    parser.add_argument('--acc_factor', type=int, help='Acceleration factor for Fourier undersampling.'
                                                       '(NOTE) only used for retrospective model!', default=4)
    parser.add_argument('--center_fraction', type=float, help='Fraction of ACS region to keep.'
                                                       '(NOTE) only used for retrospective model!', default=0.08)
    parser.add_argument('--save_dir', default='./results')
    parser.add_argument('--N', type=int, help='Number of iterations for score-POCS sampling', default=500)
    parser.add_argument('--m', type=int, help='Number of corrector step per single predictor step.'
                                              'It is advised not to change this default value.', default=1)
    return parser


def psnr(gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)


#Convert image so there are two channels for real and imaginary and c coils
def chans_to_coils(multicoil_imgs, training= False):
    
    #Add a batch dimension if needed
    if len(multicoil_imgs.shape) < 4:
        multicoil_imgs = multicoil_imgs.unsqueeze(0)
        
    b, c, h, w = multicoil_imgs.shape
        
    #Split into real and imag
    #if training:
    #    multicoil_imgs = multicoil_imgs.reshape(b, -1, 2, h, w)
    #    multicoil_imgs = multicoil_imgs.permute(0,1,3,4,2)#.contiguous()
    #else:
    multicoil_imgs = torch.stack([multicoil_imgs[:,0:int(c):2,:,:], multicoil_imgs[:,1:int(c):2,:,:]], dim=-1)

    
    return multicoil_imgs

def show_multicoil_combo(multicoil_imgs, maps, val_range = None, return_grid=False, **kwargs):
    
    #Get the coil images from the channels
    if multicoil_imgs.shape[-1] != 2:
        multicoil_imgs = chans_to_coils(multicoil_imgs)
    
    b, num_coils, h, w, _ = multicoil_imgs.shape
    
    if torch.is_tensor(maps):
        if len(maps.shape) <4:
            maps = maps.unsqueeze(0)
        maps = maps.cpu().numpy()
    else:
        if len(maps.shape)<4:
            maps = np.expand_dims(maps, axis=0)
        
    combo_imgs = []
    for i in range(b):
        with sp.Device(0):
            #Show SENSE estimate
            S = sp.linop.Multiply((h,w), maps[i])
            combo_img = S.H * tensor_to_complex_np(multicoil_imgs[i].cpu())
            
            combo_imgs.append(to_tensor(combo_img))
        
    combo_imgs = torch.stack(combo_imgs)
    
    if val_range is None:
        
        if return_grid:
            grid = show_img(combo_imgs.cpu(), return_grid=True, **kwargs)
        
        else: 
            show_img(combo_imgs.cpu(), return_grid=False, **kwargs)
            
    else:
        if return_grid:
            grid = show_img(combo_imgs.cpu(), val_range = val_range, return_grid=True, **kwargs)
        
        else: 
            show_img(combo_imgs.cpu(), val_range = val_range, return_grid=False, **kwargs)
    
    if return_grid:
        return grid
    
def show_img(plot_imgs, nrow=5, return_grid=False, mean = 0, std = 1, colormap = None, scale_each = True, **kwargs):
    '''
    Show any type of image (real or complex)
    '''
    
    #Check if it is a np array
    if isinstance(plot_imgs, np.ndarray):
        plot_imgs = torch.from_numpy(plot_imgs)
        
    #Put the images on the cpu
    plot_imgs = plot_imgs.detach().cpu()
    
    #Make the image 4 dims
    if len(plot_imgs.shape) == 3:
        plot_imgs = plot_imgs.unsqueeze(0)
        
    #Put the channel dimensions in the second dimension if needed
    if plot_imgs.shape[-1] < 5:
        plot_imgs = plot_imgs.permute(0,3,1,2)
        
    #Get the magnitude image if complex
    if plot_imgs.shape[1] == 2:
        plot_imgs = plot_imgs.permute(0,2,3,1)
        #plot_imgs = fastmri.complex_abs(plot_imgs*(std + 1e-11) + mean).unsqueeze(1)
        plot_imgs = fastmri.complex_abs(plot_imgs).unsqueeze(1)
        
        
    #Put the images in a grid and show them
    if (plot_imgs[0].dtype == torch.int32) or (plot_imgs[0].dtype == torch.uint8):
        grid = torchvision.utils.make_grid(plot_imgs, nrow = int(nrow), scale_each=False, normalize=False)
        
    elif 'val_range' in kwargs:
        grid = torchvision.utils.make_grid(plot_imgs, nrow = int(nrow), normalize=True, value_range=kwargs['val_range'])
        
    elif colormap is not None:
        grid = torchvision.utils.make_grid(plot_imgs, nrow = int(nrow), scale_each=False, normalize=False)
        
    else:
        grid = torchvision.utils.make_grid(plot_imgs, nrow = int(nrow), scale_each=scale_each, normalize=True)
        
    
        
    if 'save_dir' in kwargs:
        
        if 'rect' in kwargs:
            f = plt.figure()
            f.set_figheight(15)
            f.set_figwidth(15)
            
            if 'rect' in kwargs:
                for i, rect in enumerate(kwargs['rect']):
                    plt.gca().add_patch(Rectangle((rect[0],rect[1]),rect[2],rect[3],linewidth=5,edgecolor=kwargs['rect_colors'][i],facecolor='none'))        
                #grid = draw_bounding_boxes(grid, kwargs['rect'], colors=kwargs['rect_colors'])
                
            plt.axis("off")
            
            if colormap is not None:
                plt.imshow(grid[0].unsqueeze(0).permute(1, 2, 0).numpy(), cmap='RdGy', vmin=grid.min(), vmax=grid.max())
            else:
                plt.imshow(grid.permute(1, 2, 0).numpy())
        
            #Use a custom title
            if 'title' in kwargs:
                plt.title(kwargs['title'])
                
            #Save the image
            plt.savefig(kwargs['save_dir'], bbox_inches='tight',pad_inches=0)
        
        else:
            #Save the image
            torchvision.utils.save_image(grid, kwargs['save_dir'])
    
    else:
        if not return_grid:
            f = plt.figure()
            f.set_figheight(15)
            f.set_figwidth(15)
            
            if 'rect' in kwargs:
                for i, rect in enumerate(kwargs['rect']):
                    plt.gca().add_patch(Rectangle((rect[0],rect[1]),rect[2],rect[3],linewidth=2,edgecolor=kwargs['rect_colors'][i],facecolor='none'))        
                #grid = draw_bounding_boxes(grid, kwargs['rect'], colors=kwargs['rect_colors'])
                
            plt.axis("off")
            
            if colormap is not None:
                plt.imshow(grid[0].unsqueeze(0).permute(1, 2, 0).numpy(), cmap='RdGy', vmin=grid.min(), vmax=grid.max())
            else:
                plt.imshow(grid.permute(1, 2, 0).numpy())
        
            #Use a custom title
            if 'title' in kwargs:
                plt.title(kwargs['title'])
                
        else:
            return grid


if __name__ == "__main__":
    ###############################################
    # 1. Configurations
    ###############################################

    # args
    N = 200
    m = 1
    fname = 'val25'
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
        mps = mr.app.EspiritCalib(under_kspace.cpu().detach().squeeze().numpy(),
                                  calib_width = num_acs,
                                  crop=0.7,
                                  device=sp.Device(0),
                                  kernel_width=6).run()
        np.save(str(save_root / f'sens.npy'), mps)
    mps = torch.from_numpy(mps).view(1, num_coils, 320, 320).to(kspace.device)

    #Defines the reconstruction procedure
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