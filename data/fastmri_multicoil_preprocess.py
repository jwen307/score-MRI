#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 06:56:12 2022

@author: jeff
"""


import torch
import pytorch_lightning as pl
import os
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm

from typing import Dict, Optional, Sequence, Tuple, Union
import time

import sigpy as sp
import sigpy.mri as mr

import sys
sys.path.append("..")

import fastmri
from fastmri.data.transforms import to_tensor, tensor_to_complex_np
from datasets.fastmri_dataset import FastMRIDataModule
from datasets.fastmri_preprocess_vol_gro import retrieve_metadata, et_query, fetch_dir
from util import util, viz, network_utils, mail
from masks.mask import get_mask, apply_mask

# THIS IS FOR COMPRESSING COILS AND CROPPING - I CAN SEND THE PERTINENT PAPER IF YOU WISH
# Needs to be [imgsize, imgsize, num_coils]
def ImageCropandKspaceCompression(x, size, num_vcoils = 8):
    w_from = (x.shape[0] - size) // 2  # crop images into 384x384
    h_from = (x.shape[1] - size) // 2
    w_to = w_from + size
    h_to = h_from + size
    cropped_x = x[w_from:w_to, h_from:h_to, :]
    if cropped_x.shape[-1] > num_vcoils:
        x_tocompression = cropped_x.reshape(size ** 2, cropped_x.shape[-1])
        U, S, Vh = np.linalg.svd(x_tocompression, full_matrices=False)
        coil_compressed_x = np.matmul(x_tocompression, Vh.conj().T)
        coil_compressed_x = coil_compressed_x[:, 0:num_vcoils].reshape(size, size, num_vcoils)
    else:
        coil_compressed_x = cropped_x

    return coil_compressed_x


def ImageCropandKspaceCompressionGPU(x, size):
    w_from = (x.shape[0] - size) // 2  # crop images into 384x384
    h_from = (x.shape[1] - size) // 2
    w_to = w_from + size
    h_to = h_from + size
    cropped_x = x[w_from:w_to, h_from:h_to, :]
    if cropped_x.shape[-1] > 8:
        x_tocompression = cropped_x.reshape(size ** 2, cropped_x.shape[-1])
        U, S, Vh = torch.linalg.svd(x_tocompression, full_matrices=False)
        coil_compressed_x = torch.matmul(x_tocompression, Vh.resolve_conj().T)
        coil_compressed_x = coil_compressed_x[:, 0:8].reshape(size, size, 8)
    else:
        coil_compressed_x = cropped_x

    return coil_compressed_x

def get_compressed(kspace: np.ndarray, img_size, mask_type, accel_rate, attrs: Dict, fname:str, num_vcoils = 8):
    #kspace is dimension [num_slices, num_coils, size0, size1]
    kspace_torch = to_tensor(kspace[:])

    # check for max value
    max_value = attrs["max"] if "max" in attrs.keys() else 0.0
    acquisition = attrs["acquisition"]
    
    zf_imgs = []
    zf_mags = []
    gt_imgs = []
    slice_nums = []
    compressed_imgs = []
    
    # inverse Fourier transform to get gt solution
    gt_img = fastmri.ifft2c(kspace_torch)

    #start = time.time()
    for i in range(kspace_torch.shape[0]):
        #Compress to 8 virtual coils and crop
        compressed_img = ImageCropandKspaceCompression(tensor_to_complex_np(gt_img[i]).transpose(1,2,0),
                                                       img_size, num_vcoils)
        compressed_imgs.append(to_tensor(compressed_img))
    #end = time.time()
    #print(end-start)
    compressed_imgs = torch.stack(compressed_imgs)
        
    '''
    compressed_imgs = []
    start = time.time()
    for i in range(kspace_torch.shape[0]):
        #Compress to 8 virtual coils and crop
        compressed_img = ImageCropandKspaceCompressionGPU(torch.view_as_complex(gt_img[i]).permute(1,2,0).cuda(),
                                                       img_size)
        compressed_imgs.append(to_tensor(compressed_img.cpu()))
    end = time.time()
    print(end-start)
        
    compressed_imgs_gpu = torch.stack(compressed_imgs)
    '''
        
    #Get the kspace for compressed imgs
    compressed_k = fastmri.fft2c(compressed_imgs.permute(0,3,1,2,4))
    
    return compressed_k

        
def get_normalizing_val(kspace: np.ndarray, img_size, mask_type, accel_rate):
    """
    Args:
        kspace: Input k-space of shape (num_coils, rows, cols) for
            multi-coil data or (rows, cols) for single coil data.
        mask: Mask from the test dataset.
        target: Target image.
        attrs: Acquisition related information stored in the HDF5 object.
        fname: File name.
        slice_num: Serial number of the slice.
    Returns:
        A tuple containing, zero-filled input image, the reconstruction
        target, the mean used for normalization, the standard deviations
        used for normalization, the filename, and the slice number.
    """

    #kspace is dimension [num_slices, num_coils, size0, size1]
    kspace_torch = to_tensor(kspace)
    
    #Apply the mask
    mask = get_mask(accel=accel_rate, size=img_size, mask_type=mask_type)
    masked_kspace = apply_mask(kspace_torch, mask)
    
    #Get the zf imgs
    masked_imgs = fastmri.ifft2c(masked_kspace)
    
    #Get the magnitude imgs for the zf imgs
    zf_mags = fastmri.complex_abs(masked_imgs)
    '''
    for i in range(kspace_torch.shape[0]):

        # inverse Fourier transform to get gt solution
        gt_img = fastmri.ifft2c(kspace_torch[i])
        
        #Compress to 8 virtual coils and crop
        compressed_img = ImageCropandKspaceCompression(tensor_to_complex_np(gt_img).transpose(1,2,0),
                                                       img_size)
        
        compressed_img = to_tensor(compressed_img)
        #gt_imgs.append(compressed_img.reshape(self.img_size, self.img_size, -1).permute(2,0,1).unsqueeze(0))
        
        
        #Get the kspace for compressed imgs
        compressed_k = fastmri.fft2c(compressed_img.permute(2,0,1,3))
        
        #Apply the mask
        mask = get_mask(accel=accel_rate, size=img_size, mask_type=mask_type)
        masked_kspace = apply_mask(compressed_k, mask)
        
        #Get the zf imgs
        masked_imgs = fastmri.ifft2c(masked_kspace)
        #zf_imgs.append(masked_imgs.permute(0,3,1,2).reshape(-1,self.img_size,self.img_size).unsqueeze(0))

        #Get the magnitude imgs for the zf imgs
        zf_mags.append(fastmri.complex_abs(masked_imgs).unsqueeze(0))
        
        #slice_nums.append(i)
    '''

    #Concatenate all the slices to one tensor
    #zf_imgs = torch.cat(zf_imgs, dim=0)
    #gt_imgs = torch.cat(gt_imgs, dim=0)
    #zf_mags = torch.cat(zf_mags, dim=0)
    
    #Normalized based on the 95th percentile max value of the magnitude
    max_val = np.percentile(zf_mags, 95)
    #zf_imgs = zf_imgs/max_val
    #gt_imgs = zf_imgs/max_val
    

    return masked_kspace, max_val
    #return zf_imgs, gt_imgs, mask, fname, slice_nums, acquisition



# Location of the dataset
base_dir = "/storage/fastMRI/data/"
#base_dir= '/scratch/fastMRI/data/'
#base_dir = '/storage/data/'
#base_dir = '../../datasets/fastMRI/data'

#Parameters
accel_rate = 4
img_size = 320
num_vcoils = 2
dataset_type = 'train'
challenge = 'multicoil'
mask_type = 's4'
dataset_dir = os.path.join(base_dir, '{0}_{1}'.format(challenge, dataset_type))
new_dir = os.path.join(base_dir, '{0}_{1}_{2}coils'.format(challenge, dataset_type, num_vcoils))






#Determine which challenge is being used
if challenge == 'singlecoil':
    recons_key = ('reconstruction_esc')
else:
    recons_key = ('reconstruction_rss')
    
#Create the directory if it does not already exist
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

#%% Get the compressed coil images

files = list(Path(dataset_dir).iterdir())

for fname in tqdm(sorted(files)):
    
    #Recover the metadata
    metadata, num_slices = retrieve_metadata(fname)

    with h5py.File(fname, "r") as hf:
        
        #Get the kspace
        kspace = hf['kspace']
        
        #Get the attributes of the volume
        attrs = dict(hf.attrs)
        attrs.update(metadata)
        
        #Get the virtual coils and the normalization value for the volume
        compressed_k = get_compressed(kspace, img_size, mask_type, accel_rate, attrs, fname.name, num_vcoils)
    
    
    #Save the processed data into the new h5py file
    with h5py.File(os.path.join(new_dir, fname.name), 'w') as nf:
        target_k = nf.create_dataset('target_kspace', compressed_k.numpy().shape, data=compressed_k.numpy())
        nf.attrs['num_slices'] = num_slices
        nf.attrs['mask_type'] = mask_type
        nf.attrs['acquisition'] = attrs["acquisition"]
        nf.attrs['fname'] = fname.name
  
    

#%% Get the normalizing value
max_val_dir = os.path.join(base_dir, '{0}_maxval_{2}coils_{1}'.format(dataset_type,mask_type, num_vcoils))

#Create the directory if it does not already exist
if not os.path.exists(max_val_dir):
    os.makedirs(max_val_dir)

files = list(Path(new_dir).iterdir())

for fname in tqdm(sorted(files)):
    
    with h5py.File(fname, "r") as hf:
        #Get the compressed target kspace
        kspace = hf['target_kspace'][:]
        
        #Find the normalizing value
        masked_kspace, max_val = get_normalizing_val(kspace, img_size, mask_type, accel_rate)
        
        sense_maps = []
        
        for mk in masked_kspace:
            #Find the sensitivity maps
            maps = mr.app.EspiritCalib(tensor_to_complex_np(mk), 
                                       calib_width=13,
                                       show_pbar=False, 
                                       crop=0.70, 
                                       device=sp.Device(0),
                                       kernel_width=6).run().get()
            
            sense_maps.append(maps)
            
        sense_maps = np.stack(sense_maps)
    
    with h5py.File(os.path.join(max_val_dir, fname.name), 'w') as nf:
        nf.attrs['max_val'] = max_val
        sense_maps_nf = nf.create_dataset('sense_maps', sense_maps.shape, data=sense_maps)
        