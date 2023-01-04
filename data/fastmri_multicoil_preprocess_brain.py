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
import cupy as cp

import sys
sys.path.append("..")

import fastmri
from fastmri.data.transforms import to_tensor, tensor_to_complex_np
from datasets.fastmri_dataset import FastMRIDataModule
from datasets.fastmri_preprocess_vol_gro import retrieve_metadata, et_query, fetch_dir
from util import util, viz, network_utils, mail
from datasets.masks.mask import get_mask, apply_mask

# Coil compression and image cropping
# Needs to be [imgsize, imgsize, num_coils]
# Modified so it uses cupy and is 10x faster
def ImageCropandKspaceCompression(x, size, num_vcoils = 8, vh = None):
    w_from = (x.shape[0] - size) // 2  # crop images into 384x384
    h_from = (x.shape[1] - size) // 2
    w_to = w_from + size
    h_to = h_from + size
    cropped_x = x[w_from:w_to, h_from:h_to, :]
    if cropped_x.shape[-1] > num_vcoils:
        x_tocompression = cropped_x.reshape(size ** 2, cropped_x.shape[-1])
        
        if vh is None:
            #Convert to a cupy tensor
            with cp.cuda.Device(0):
                x_tocompression = cp.asarray(x_tocompression)
                U, S, Vh = cp.linalg.svd(x_tocompression, full_matrices=False)
                coil_compressed_x = cp.matmul(x_tocompression, Vh.conj().T)
                coil_compressed_x = coil_compressed_x[:, 0:num_vcoils].reshape(size, size, num_vcoils)
            # x_tocompression = np.asarray(x_tocompression)
            # U, S, Vh = np.linalg.svd(x_tocompression, full_matrices=False)
            # coil_compressed_x = np.matmul(x_tocompression, Vh.conj().T)
            # coil_compressed_x = coil_compressed_x[:, 0:num_vcoils].reshape(size, size, num_vcoils)
                
        else:
            coil_compressed_x = np.matmul(x_tocompression, vh.conj().T)
            coil_compressed_x = coil_compressed_x[:, 0:num_vcoils].reshape(size, size, num_vcoils)
            Vh = vh
                
    else:
        coil_compressed_x = cropped_x

    if vh is not None:
        return coil_compressed_x
    return cp.asnumpy(coil_compressed_x), cp.asnumpy(Vh)


def get_compressed(kspace: np.ndarray, img_size, mask_type, accel_rate, attrs: Dict, fname:str, num_vcoils = 8, vh=None):
    #kspace is dimension [num_slices, num_coils, size0, size1]


    # inverse Fourier transform to get gt solution
    gt_img = fastmri.ifft2c(kspace)
    
    compressed_imgs = []
    Vhs = []

    for i in range(kspace_torch.shape[0]):
        #Compress to 8 virtual coils and crop
        compressed_img, Vh = ImageCropandKspaceCompression(tensor_to_complex_np(gt_img[i]).transpose(1,2,0),
                                                       img_size, num_vcoils, vh)
        compressed_imgs.append(to_tensor(compressed_img))
        Vhs.append(Vh)

    #Combine into one tensor stack
    compressed_imgs = torch.stack(compressed_imgs)
    Vhs = np.stack(Vhs)
    

        
    #Get the kspace for compressed imgs
    compressed_k = fastmri.fft2c(compressed_imgs.permute(0,3,1,2,4))
    
    if vh is None:
        return compressed_k, Vhs
    else:
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
    #kspace_torch = to_tensor(kspace)
    
    #Apply the mask
    mask = get_mask(accel=accel_rate, size=img_size, mask_type=mask_type)
    masked_kspace = apply_mask(kspace, mask)
    
    #Get the zf imgs
    masked_imgs = fastmri.ifft2c(masked_kspace)
    
    #Get the magnitude imgs for the zf imgs
    zf_mags = fastmri.complex_abs(masked_imgs)


    #Normalized based on the 95th percentile max value of the magnitude
    max_val = np.percentile(zf_mags.cpu(), 95)
    

    return masked_kspace, max_val
    #return zf_imgs, gt_imgs, mask, fname, slice_nums, acquisition


if __name__ == '__main__':

    # Location of the dataset
    base_dir = "/storage/fastMRI_brain/data/"
    #base_dir= '/scratch/fastMRI/data/'
    #base_dir = '/storage/data/'
    #base_dir = '../../datasets/fastMRI/data'
    
    #Parameters
    accel_rate = 4
    img_size = 384
    num_vcoils = 8
    dataset_type = 'train'
    challenge = 'multicoil'
    mask_type = 'matt'
    #dataset_dir = os.path.join(base_dir, '{0}_{1}'.format(challenge, dataset_type))
    dataset_dir = os.path.join(base_dir, 'small_T2_test')
    #new_dir = os.path.join(base_dir, '{0}_{1}_{2}coils'.format(challenge, dataset_type, num_vcoils))
    new_dir = os.path.join(base_dir, '{0}_{1}_{2}coils'.format(challenge, 'small_T2_test', num_vcoils))
    
    
    #Create the directory if it does not already exist
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    
    #%% Get the compressed coil images
    
    files = list(Path(dataset_dir).iterdir())
    
    for fname in tqdm(sorted(files)):
        
        #Skip non-data files
        if fname.name[0] == '.':
            continue
        
        #Recover the metadata
        metadata, num_slices = retrieve_metadata(fname)
    
        with h5py.File(fname, "r") as hf:
            
            #Get the kspace
            kspace = hf['kspace']
            
            #Get the attributes of the volume
            attrs = dict(hf.attrs)
            attrs.update(metadata)
            
            if attrs['acquisition'] != 'AXT2' or attrs['encoding_size'][1] < 384:
                continue
    
            kspace_torch = to_tensor(kspace[:])
            
            if kspace_torch.shape[1] <= num_vcoils:
                continue


            #Get the virtual coils and the normalization value for the volume
            #start = time.time()
            compressed_k, vhs = get_compressed(kspace_torch, img_size, mask_type, accel_rate, attrs, fname.name, num_vcoils)
            #end = time.time()
            masked_kspace, max_val = get_normalizing_val(compressed_k, img_size, mask_type, accel_rate)
            #end2 = time.time()
            #print(end-start)
            #print(end2-end)



        #Save the processed data into the new h5py file
        with h5py.File(os.path.join(new_dir, fname.name), 'w') as nf:
            nf.attrs['max_val'] = max_val
            #Save the Vh from the svd
            vh = nf.create_dataset('vh', vhs.shape, data=vhs)
        

  
#%% Test how fast the coil compression is
'''

compressed_imgs = []

# inverse Fourier transform to get gt solution
gt_img = fastmri.ifft2c(kspace_torch)

start = time.time()
for i in range(kspace_torch.shape[0]):
    #Compress to 8 virtual coils and crop
    compressed_img = ImageCropandKspaceCompression(tensor_to_complex_np(gt_img[i]).transpose(1,2,0),
                                                   img_size, num_vcoils)
    compressed_imgs.append(to_tensor(compressed_img))
end = time.time()
print(end-start)
compressed_imgs_cpu = torch.stack(compressed_imgs)
    

# compressed_imgs = []
# start = time.time()
# for i in range(kspace_torch.shape[0]):
#     #Compress to 8 virtual coils and crop
#     compressed_img = ImageCropandKspaceCompressionGPU(torch.view_as_complex(gt_img[i]).permute(1,2,0).cuda(),
#                                                    img_size)
#     compressed_imgs.append(to_tensor(compressed_img.cpu()))
# end = time.time()
# print(end-start)
    
# compressed_imgs_gpu = torch.stack(compressed_imgs)


#%%
import cupy
x = tensor_to_complex_np(gt_img[0]).transpose(1,2,0)
size = 384
w_from = (x.shape[0] - size) // 2  # crop images into 384x384
h_from = (x.shape[1] - size) // 2
w_to = w_from + size
h_to = h_from + size
cropped_x = x[w_from:w_to, h_from:h_to, :]

x_tocompression = cropped_x.reshape(size ** 2, cropped_x.shape[-1])
U, S, Vh = cupy.linalg.svd(cupy.asarray(x_tocompression), full_matrices=False)
coil_compressed_x = np.matmul(cupy.asnumpy(x_tocompression), cupy.asnumpy(Vh.conj().T))
coil_compressed_x = coil_compressed_x[:, 0:num_vcoils].reshape(size, size, num_vcoils)
full_imgs = to_tensor(coil_compressed_x).permute(2,0,1,3)
#Get the kspace for compressed imgs
compressed_k = fastmri.fft2c(to_tensor(coil_compressed_x).permute(2,0,1,3))


#Apply the mask
mask = get_mask(accel=accel_rate, size=img_size, mask_type=mask_type)
masked_kspace = apply_mask(compressed_k, mask)

#Get the zf imgs
masked_imgs = fastmri.ifft2c(masked_kspace)

#Get the magnitude imgs for the zf imgs
zf_mags = fastmri.complex_abs(masked_imgs)

#Normalized based on the 95th percentile max value of the magnitude
max_val = np.percentile(zf_mags, 95)

#Find the sensitivity maps
start = time.time()
maps = mr.app.EspiritCalib(tensor_to_complex_np(masked_kspace), 
                            calib_width=31,
                            show_pbar=False, 
                            crop=0.70, 
                            device=sp.Device(0),
                            kernel_width=6).run().get()
end = time.time()
print(end-start)

viz.show_multicoil_combo(full_imgs.unsqueeze(0), maps)
viz.show_multicoil_combo(masked_imgs.unsqueeze(0), maps)


#What you would get if you found the normalizing value from the single-coil magnitude image
single_coil = network_utils.multicoil2single(masked_imgs.unsqueeze(0), maps)
mag = fastmri.complex_abs(single_coil)
max_val1 = np.percentile(mag,95)


x = torch.view_as_complex(gt_img[0]).permute(1,2,0).cuda()
size = 384
w_from = (x.shape[0] - size) // 2  # crop images into 384x384
h_from = (x.shape[1] - size) // 2
w_to = w_from + size
h_to = h_from + size
cropped_xt = x[w_from:w_to, h_from:h_to, :]

x_tocompressiont = cropped_xt.reshape(size ** 2, cropped_x.shape[-1])
Ut, St, Vht = torch.linalg.svd(x_tocompressiont, full_matrices=False)
coil_compressed_xt = torch.matmul(x_tocompressiont, Vht.resolve_conj().T)
coil_compressed_xt = coil_compressed_xt[:, 0:8].reshape(size, size, 8)


full_imgst = to_tensor(coil_compressed_xt.cpu()).permute(2,0,1,3)
#Get the kspace for compressed imgs
compressed_kt = fastmri.fft2c(full_imgst)


#Apply the mask
mask = get_mask(accel=accel_rate, size=img_size, mask_type=mask_type)
masked_kspacet = apply_mask(compressed_kt, mask)

#Get the zf imgs
masked_imgst = fastmri.ifft2c(masked_kspacet)

#Get the magnitude imgs for the zf imgs
zf_magst = fastmri.complex_abs(masked_imgst)

#Normalized based on the 95th percentile max value of the magnitude
max_valt = np.percentile(zf_magst, 95)

#Find the sensitivity maps
start = time.time()
maps = mr.app.EspiritCalib(tensor_to_complex_np(masked_kspacet), 
                            calib_width=31,
                            show_pbar=False, 
                            crop=0.70, 
                            device=sp.Device(0),
                            kernel_width=6).run().get()
end = time.time()
print(end-start)

viz.show_multicoil_combo(full_imgst.unsqueeze(0), maps)
viz.show_multicoil_combo(masked_imgst.unsqueeze(0), maps)


#What you would get if you found the normalizing value from the single-coil magnitude image
single_coilt = network_utils.multicoil2single(masked_imgst.unsqueeze(0), maps)
magt = fastmri.complex_abs(single_coilt)
max_val1t = np.percentile(magt,95)


#%% See what happens when you combined the two


#%% Get the normalizing value
# max_val_dir = os.path.join(base_dir, '{0}_maxval_{2}coils_{1}'.format(dataset_type,mask_type, num_vcoils))

# #Create the directory if it does not already exist
# if not os.path.exists(max_val_dir):
#     os.makedirs(max_val_dir)

# files = list(Path(new_dir).iterdir())

# for fname in tqdm(sorted(files)):
    
#     with h5py.File(fname, "r") as hf:
#         #Get the compressed target kspace
#         kspace = hf['target_kspace'][:]
        
#         #Find the normalizing value
#         masked_kspace, max_val = get_normalizing_val(kspace, img_size, mask_type, accel_rate)
        
#         sense_maps = []
        
#         for mk in masked_kspace:
#             #Find the sensitivity maps
#             maps = mr.app.EspiritCalib(tensor_to_complex_np(mk), 
#                                        calib_width=13,
#                                        show_pbar=False, 
#                                        crop=0.70, 
#                                        device=sp.Device(0),
#                                        kernel_width=6).run().get()
            
#             sense_maps.append(maps)
            
#         sense_maps = np.stack(sense_maps)
    
#     with h5py.File(os.path.join(max_val_dir, fname.name), 'w') as nf:
#         nf.attrs['max_val'] = max_val
#         sense_maps_nf = nf.create_dataset('sense_maps', sense_maps.shape, data=sense_maps)
'''
        