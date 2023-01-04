import os
from pathlib import Path

import yaml

from util import network_utils
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
from data.fastmri_multicoil import FastMRIDataModule
from tqdm import tqdm

import utils


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

def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Mean Squared Error (MSE)"""
    return np.mean((gt - pred) ** 2)


def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2

def rsnr(gt: np.ndarray, pred: np.ndarray):
    # Do RSNR not in dB (convert to dB after finding the mean)
    # rsnr = np.linalg.norm(gt)**2 / np.linalg.norm(gt - pred)**2

    # Do it in dB
    rsnr = 10 * np.log10(np.linalg.norm(gt) ** 2 / np.linalg.norm(gt - pred) ** 2)

    return rsnr


def calc_metrics(reconstructions, targets, is_complex=False):
    nmses = []
    psnrs = []
    ssims = []
    rsnrs = []
    for i in tqdm(range(len(reconstructions))):
        nmses.append(nmse(targets[i], reconstructions[i]))
        psnrs.append(psnr(targets[i], reconstructions[i]))
        ssims.append(ssim(targets[i], reconstructions[i]))
        rsnrs.append(rsnr(targets[i], reconstructions[i]))

    report_dict = {
        'results': {
            'mean_nmse': float(np.mean(nmses)),
            'std_err_nmse': float(np.std(nmses) / np.sqrt(len(nmses))),
            'mean_psnr': float(np.mean(psnrs)),
            'std_err_psnr': float(np.std(psnrs) / np.sqrt(len(psnrs))),
            'mean_ssim': float(np.mean(ssims)),
            'std_err_ssim': float(np.std(ssims) / np.sqrt(len(ssims))),
            'mean_rsnr (db)': float(np.mean(rsnrs)),
            'std_err_rsnr (db)': float(np.std(rsnrs) / np.sqrt(len(rsnrs))),
        }
    }

    return report_dict


# %%
if __name__ == "__main__":
    ###############################################
    # 1. Configurations
    ###############################################

    # args
    N = 200
    m = 1


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


#%%
    ###############################################
    # 2. Inference
    ###############################################
    #Setup the data
    kwargs = {'center_frac': 0.04,
              'accel_rate': 4,
              'img_size': 320,
              'complex': True,
              'challenge': 'multicoil',
              'vol': True,
              'img_type': 'mag',
              'mask_type': 's4',
              'scan_type': 'CORPD_FBK',
              'num_vcoils': 8
              }
    
    base_dir = "/storage/fastMRI/data/"
    data = FastMRIDataModule(base_dir, batch = 16, **kwargs)
    data.prepare_data()
    data.setup()
    
    #Pick 72 random indices
    #idxs = torch.randperm(len(data.val))[0:72].numpy()
    
    idxs = np.load(str(save_root / 'recon' / '72_eval_idxs.npy'))
    
    dataset = data.val
    
    sample_psnrs = []
    mean_psnrs = []
    sample_ssims = []
    mean_ssims = []
    
    all_samples = []
    
    # Get multiple posterior samples
    print(f'Beginning inference')
    tic = time.time()
    
    for k, samp_num in tqdm(enumerate(idxs)):

        # Get the data
        cond = dataset[samp_num][0]
        gt = dataset[samp_num][1]
        #maps = dataset[samp_num][2]
        #maps = network_utils.get_maps(cond, model.acs_size)
        mask = dataset[samp_num][3]
        std = torch.tensor(dataset[samp_num][4])
        
        
        #Convert to a complex np array
        gt = network_utils.chans_to_coils(gt) * std
        gt = fastmri.tensor_to_complex_np(gt)
        
        #Get a 2D mask
        mask = mask.permute(0,2,1).repeat(1,mask.shape[1],1)
    
        # Read data
        gt = normalize_complex(torch.from_numpy(gt.astype(np.complex64)))
        # TODO: Changed to 8 virtual coils (JW)
        gt = gt.view(1, num_coils, 320, 320)
        gt = gt.to(config.device)
    
        # TODO: Changed to look for retrospective and prospective (JW)
        mask = mask.view(1, 1, 320, 320).to(config.device)
        
        # fft
        kspace = fft2_m(gt)
    
        # undersampling
        under_kspace = kspace * mask
        under_img = ifft2_m(under_kspace)
    
        # ESPiRiT
        # ESPiRiT
        mps = mr.app.EspiritCalib(under_kspace.cpu().detach().squeeze().numpy(),
                                  calib_width=num_acs,
                                  crop=0.7,
                                  device=sp.Device(0),
                                  kernel_width=6, 
                                  show_pbar=False).run().get()
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

        
    
        #Get 8 posterior samples
        samples = []
        for i in range(8):
            #Get the samples
            x = pc_fouriercs(score_model, scaler(gt), y=under_kspace)

            #Get the singlecoil approximation
            #x = network_utils.multicoil2single(to_tensor(x.detach().cpu()), mps)
            x = to_tensor(x.detach().cpu())
    
            samples.append(x)
            
        samples = torch.stack(samples,dim=0).cpu()
        all_samples.append(samples)
        
    
    toc = time.time() - tic
    print(f'Time took for recon: {toc} secs.')
    
    
    all_samps_np = torch.stack(all_samples).cpu().numpy()
    np.save(str(save_root / 'recon' / '72_eval.npy'), all_samps_np)
    #np.save(str(save_root / 'recon' / '72_eval_idxs.npy'), idxs)
    
    
    
#%% Load the reconstructions and find the metrics
    all_samps_np = np.load(str(save_root / 'recon' / '72_eval.npy'))
    idxs = np.load(str(save_root / 'recon' / '72_eval_idxs.npy'))
    
    gts = []
    preds = []

    for k, samp_num in tqdm(enumerate(idxs)):

        # Get the data
        cond = dataset[samp_num][0]
        gt0 = dataset[samp_num][1]
        #maps = dataset[samp_num][2]
        #maps = network_utils.get_maps(cond, model.acs_size)
        mask = dataset[samp_num][3]
        std = torch.tensor(dataset[samp_num][4])
        
        #Convert to a complex np array
        gt0 = network_utils.chans_to_coils(gt0) * std
        gt = fastmri.tensor_to_complex_np(gt0)
        
        #Get a 2D mask
        mask = mask.permute(0,2,1).repeat(1,mask.shape[1],1)
    
        # Read data
        gt, min_max = utils.normalize_complex(torch.from_numpy(gt.astype(np.complex64)),give_params=True)
        # TODO: Changed to 8 virtual coils (JW)
        gt = gt.view(1, num_coils, 320, 320)
        gt = gt.to(config.device)
    
        # TODO: Changed to look for retrospective and prospective (JW)
        mask = mask.view(1, 1, 320, 320).to(config.device)
        
        # fft
        kspace = fft2_m(gt)
    
        # undersampling
        under_kspace = kspace * mask
        under_img = ifft2_m(under_kspace)
    
        # ESPiRiT
        mps = mr.app.EspiritCalib(under_kspace.cpu().detach().squeeze().numpy(),
                                  calib_width=num_acs,
                                  crop=0.7,
                                  device=sp.Device(0),
                                  kernel_width=6,
                                  show_pbar=False).run()
            #np.save(str(save_root / f'sens.npy'), mps)
        maps = torch.from_numpy(mps.get()).view(1, num_coils, 320, 320).to(kspace.device)
        #maps = dataset[samp_num][2]
        
        
        
        #Convert to a single coil
        gt = utils.unnormalize_complex(gt, min_max)
        gt = to_tensor(gt.cpu())
        #gt = network_utils.multicoil2single(gt, maps).cpu()
        gt = root_sum_of_squares(gt,dim=1)
        gt = fastmri.complex_abs(gt).numpy()
        #gt = utils.unnormalize(gt, min_max[0].numpy(), min_max[1].numpy())
        gts.append(gt)
        

        #Get the mean prediction
        pred = all_samps_np[k].mean(axis=0)
        #pred = all_samps_np[0][0]
        pred = tensor_to_complex_np(torch.tensor(pred))
        pred = utils.unnormalize_complex(torch.tensor(pred),min_max)
        pred = root_sum_of_squares(to_tensor(pred),dim=1)
        #pred = network_utils.multicoil2single(to_tensor(pred), maps).cpu()
        #pred = network_utils.multicoil2single(torch.tensor(pred), maps).cpu()
        pred = fastmri.complex_abs(torch.tensor(pred)).numpy()
        #pred = utils.unnormalize(pred, min_max[0].numpy(), min_max[1].numpy())
        preds.append(pred)
        
    report_dict = calc_metrics(preds, gts)
    
    
    report_path = os.path.join(save_root, 'psnr')
    Path(report_path).mkdir(parents=True, exist_ok=True)

    report_file_path = os.path.join(report_path, 'psnr_report{0}.yaml'.format(N))
    with open(report_file_path, 'w') as file:
        documents = yaml.dump(report_dict, file)

    print(report_dict)
        
        
            


    



















