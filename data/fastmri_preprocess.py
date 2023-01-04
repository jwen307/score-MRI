#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 09:51:20 2022

@author: jeff
"""

import os
import torch
import torchvision.transforms as transforms
import numpy as np
from fastmri.data import mri_data
from fastmri.data import subsample
from fastmri.data import subsample
from fastmri.data import transforms as mri_transforms
from fastmri import fftc
import fastmri
from typing import Dict, Optional, Sequence, Tuple, Union
import time

from pathlib import Path
import h5py
import logging
import os
import pickle
import random
import xml.etree.ElementTree as etree
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from warnings import warn

import pandas as pd
import requests

import yaml
import sys
sys.path.append('..')
from util import mri_utils, viz

    
#Data transform that will give either images or kspace
class ConditionalDataTransformNew:
    '''
    A more generic data transform for the MRI dataset
    Reformated from the fastMRI repository
    '''

    def __init__(self, mask_func: Optional[subsample.MaskFunc]=None, add_transforms = None, use_seed=True, use_complex = False, ):

        self.mask_func = mask_func
        self.use_seed = use_seed
        self.add_transforms = add_transforms
        self.use_complex = use_complex

    def __call__(self, kspace: np.ndarray, mask: np.ndarray, target: np.ndarray, attrs: Dict, fname:str, slice_num:int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
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

        kspace_torch = mri_transforms.to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0
        acquisition = attrs["acquisition"]

        # inverse Fourier transform to get gt solution
        image = fastmri.ifft2c(kspace_torch)

        # crop input to correct size
        if target is not None:
            crop_size = (target.shape[-2], target.shape[-1])
        else:
            crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        if image.shape[-2] < crop_size[1]:
            crop_size = (image.shape[-2], image.shape[-2])
            
        #Crop the image 
        gt_image = mri_transforms.complex_center_crop(image, crop_size)

        #Take the Fourier transform to get the k-space of the cropped image
        cropped_kspace = fastmri.fft2c(gt_image)
        
        
        # apply mask
        if self.mask_func:
            seed = 0 #None if not self.use_seed else tuple(map(ord, fname))
            # we only need first element, which is k-space after masking
            masked_kspace, mask = mri_transforms.apply_mask(cropped_kspace, self.mask_func, seed=seed)[0:2]
        else:
            masked_kspace = kspace_torch
            mask = None
        
        #Inverse Fourier transform to get the zero-filled image
        zf_img = fastmri.ifft2c(masked_kspace)
        
        if not self.use_complex:
            # Take the absolute value
            zf_img = fastmri.complex_abs(zf_img)
        

        # normalize 	
        zf_img, mean, std = mri_transforms.normalize_instance(zf_img, eps=1e-11)
        
        #Clamp the values if not using complex images
        if not self.use_complex: 
            zf_img = zf_img.clamp(-6, 6)
            zf_img = zf_img.reshape(1, zf_img.shape[-2], zf_img.shape[-2])
        
        else:
            zf_img = zf_img.permute(2,0,1).unsqueeze(0)
        
        #Additional transformations
        if self.add_transforms is not None:
            zf_img = self.add_transforms(zf_img)
            
        
        # normalize target
        if not self.use_complex: 
            gt_image = fastmri.complex_abs(gt_image)

        target_torch = mri_transforms.normalize(gt_image, mean, std, eps=1e-11)
        
        if not self.use_complex:
            target_torch = target_torch.clamp(-6, 6)
        
            target_torch = target_torch.reshape(1, target_torch.shape[-2], target_torch.shape[-2])
            
        else:
            target_torch = target_torch.permute(2,0,1).unsqueeze(0)
        
        #Additional transformations
        if self.add_transforms is not None:
            target_torch = self.add_transforms(target_torch)
        

        return zf_img, target_torch, masked_kspace, mask, mean, std, fname, slice_num, acquisition
        
#%% From fastMRI repository 
def et_query(
    root: etree.Element,
    qlist: Sequence[str],
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.
    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.
    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.
    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)


def fetch_dir(
    key: str, data_config_file: Union[str, Path, os.PathLike] = "fastmri_dirs.yaml"
) -> Path:
    """
    Data directory fetcher.
    This is a brute-force simple way to configure data directories for a
    project. Simply overwrite the variables for `knee_path` and `brain_path`
    and this function will retrieve the requested subsplit of the data for use.
    Args:
        key: key to retrieve path from data_config_file. Expected to be in
            ("knee_path", "brain_path", "log_path").
        data_config_file: Optional; Default path config file to fetch path
            from.
    Returns:
        The path to the specified directory.
    """
    data_config_file = Path(data_config_file)
    if not data_config_file.is_file():
        default_config = {
            "knee_path": "/path/to/knee",
            "brain_path": "/path/to/brain",
            "log_path": ".",
        }
        with open(data_config_file, "w") as f:
            yaml.dump(default_config, f)

        data_dir = default_config[key]

        warn(
            f"Path config at {data_config_file.resolve()} does not exist. "
            "A template has been created for you. "
            "Please enter the directory paths for your system to have defaults."
        )
    else:
        with open(data_config_file, "r") as f:
            data_dir = yaml.safe_load(f)[key]

    return Path(data_dir)
    
def retrieve_metadata(fname):
    with h5py.File(fname, "r") as hf:
        et_root = etree.fromstring(hf["ismrmrd_header"][()])

        enc = ["encoding", "encodedSpace", "matrixSize"]
        enc_size = (
            int(et_query(et_root, enc + ["x"])),
            int(et_query(et_root, enc + ["y"])),
            int(et_query(et_root, enc + ["z"])),
        )
        rec = ["encoding", "reconSpace", "matrixSize"]
        recon_size = (
            int(et_query(et_root, rec + ["x"])),
            int(et_query(et_root, rec + ["y"])),
            int(et_query(et_root, rec + ["z"])),
        )

        lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
        enc_limits_center = int(et_query(et_root, lims + ["center"]))
        enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

        padding_left = enc_size[1] // 2 - enc_limits_center
        padding_right = padding_left + enc_limits_max

        num_slices = hf["kspace"].shape[0]

    metadata = {
        "padding_left": padding_left,
        "padding_right": padding_right,
        "encoding_size": enc_size,
        "recon_size": recon_size,
    }

    return metadata, num_slices


def kspace2img(kspace):
    # inverse Fourier transform to get zero filled solution
    image = fastmri.ifft2c(kspace)
    
    # absolute value
    image = fastmri.complex_abs(image)

    # normalize input (actually standardizes)
    image, mean, std = mri_transforms.normalize_instance(image, eps=1e-11)
    image = image.clamp(-6, 6)
    
    return image
#%%

#Location of the dataset
base_dir = '../../datasets/fastMRI/data/'
#base_dir = '/storage/data/'
#base_dir = '/storage/fastMRI/data'
dataset_dir = os.path.join(base_dir, 'singlecoil_val')

#Parameters
center_frac = 0.08
accel_rate = 4
img_size = 320
challenge = 'singlecoil'
dataset_type = 'val'
use_complex = False

#Location of the new dataset
if not use_complex:
    new_dir = os.path.join(base_dir, '{0}_{1}_size{2}_centerfrac{3}_accel{4}'.format(challenge,dataset_type,img_size, center_frac, accel_rate))
else:
    new_dir = os.path.join(base_dir, '{0}_{1}_size{2}_centerfrac{3}_accel{4}_complex'.format(challenge,dataset_type,img_size, center_frac, accel_rate))

if __name__ == '__main__':
    
    #Create the directory if it does not already exist
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    
    #Define the mask function and the data transformation
    mask_func = subsample.RandomMaskFunc(center_fractions = [center_frac], accelerations = [accel_rate])
    resize = transforms.Resize(img_size,interpolation=transforms.functional.InterpolationMode.BILINEAR) if img_size != 320 else None
    data_transform = ConditionalDataTransformNew(mask_func = mask_func, add_transforms=resize, use_complex=use_complex)
    
    #Determine which challenge is being used
    if challenge == 'singlecoil':
        recons_key = ('reconstruction_esc')
    else:
        recons_key = ('reconstruction_rss')
    
    #Get a list of all the h5py files
    files = list(Path(dataset_dir).iterdir())
    
    #For each file, perform the transformation and save to a new file
    for fname in sorted(files):
        
        #Recover the metadata
        metadata, num_slices = retrieve_metadata(fname)
        
        #Define lists to hold all of the information for a given file
        zero_filled_imgs = []
        target_imgs = []
        masks = []
        means = []
        stds = []
        fnames = []
        slice_nums = []
        
        
        with h5py.File(fname, 'r') as hf:
            
            for dataslice in range(num_slices):
                
                #Get the kspace
                kspace = hf['kspace'][dataslice]
                
                mask = np.asarray(hf["mask"]) if "mask" in hf else None
                
                #Get the fully sampled image
                target = hf[recons_key][dataslice] if recons_key in hf else None
                
                attrs = dict(hf.attrs)
                attrs.update(metadata)
                
                #Perform the transformation
                image, target_torch, masked_kspace, mask,  mean, std, file_name, slice_num, acquisition = data_transform(kspace, mask, target, attrs, fname.name, dataslice)
                zero_filled_imgs.append(image)
                target_imgs.append(target_torch)
                masks.append(mask)
                means.append(mean)
                stds.append(std)
                slice_nums.append(slice_num)
                
                
            
            zero_filled_imgs_np = torch.cat(zero_filled_imgs, dim=0).numpy()
            target_imgs_np = torch.cat(target_imgs, dim=0).numpy()
            masks_np = torch.cat(masks, dim=0).numpy()
            means_np = np.asarray(means)
            stds_np = np.asarray(stds)
            slice_nums_np = np.asarray(slice_nums)
            
            
            #Save the processed data into the new h5py file
            with h5py.File(os.path.join(new_dir, fname.name), 'w') as nf:
                
                zf_imgs = nf.create_dataset('zero_filled_imgs', zero_filled_imgs_np.shape, data=zero_filled_imgs_np)
                
                t_imgs = nf.create_dataset('target_imgs', target_imgs_np.shape, data=target_imgs_np)
                
                masks_ds = nf.create_dataset('mask', masks_np.shape, data=masks_np)
                
                means_ds = nf.create_dataset('means', means_np.shape, data=means_np)
                
                stds_ds = nf.create_dataset('stds', stds_np.shape, data=stds_np)

                nf.attrs['acquisition'] = acquisition
                
                nf.attrs['fname'] = file_name
                
                slice_nums_ds = nf.create_dataset('slice_nums', slice_nums_np.shape, data=slice_nums_np)
                
                
         
                
                
    '''
    mask_func = subsample.RandomMaskFunc(center_fractions = [center_frac], accelerations = [accel_rate])
    #data_transform = ConditionalDataTransformNew(mask_func = mask_func, add_transforms=transforms.Resize(img_size,interpolation=transforms.functional.InterpolationMode.BILINEAR))
    data_transform = ConditionalDataTransformNew(mask_func = mask_func, use_complex=True)
    
    #Load the data 
    trainset = mri_data.SliceDataset(root = dataset_dir, transform = data_transform, challenge='singlecoil')
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 8, shuffle = True, num_workers = 12, pin_memory=False)
    
    start = time.time()
    #for i, data in enumerate(trainloader):
    for i in range(100)
        test = trainset[i]
    
    end = time.time()
    print(end-start)
    '''
          
#%%
    '''
    zf_kspace = fastmri.fft2c(torch.cat([image, torch.zeros_like(image)], dim=0).permute(1,2,0))
    
    target_kspace = fastmri.fft2c(torch.cat([target_torch, torch.zeros_like(target_torch)], dim=0).permute(1,2,0))
    
    target2 = fastmri.ifft2c(masked_kspace)
    target2 = fastmri.complex_abs(target2).unsqueeze(0)
    target2_kspace = fastmri.fft2c(torch.cat([target2, torch.zeros_like(target2)], dim=0).permute(1,2,0))
    '''
    
#%%
    '''
    rand = [np.random.randint(0,len(trainset)) for _ in range(20)]
    for i in rand:
        kspace_torch = mri_transforms.to_tensor(trainset[i][0])
        viz.show_tensor_imgs(kspace2img(kspace_torch))
        viz.show_tensor_imgs(torch.tensor(trainset[i][2]))
    '''
