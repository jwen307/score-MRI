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
from torch.utils.data import DataLoader

from typing import Dict, Optional, Sequence, Tuple, Union

import sigpy as sp
import sigpy.mri as mr

import sys
sys.path.append("..")

import fastmri
from fastmri.data.transforms import to_tensor, tensor_to_complex_np
from datasets.fastmri_dataset import FastMRIDataModule
from datasets.fastmri_preprocess_vol_gro import retrieve_metadata, et_query, fetch_dir
from util import util, viz, network_utils, mail
from datasets.masks.mask import get_mask, apply_mask
from datasets.fastmri_multicoil_preprocess_brain import ImageCropandKspaceCompression

import time


def get_compressed(kspace: np.ndarray, img_size, num_vcoils = 8, vh = None):
    # inverse Fourier transform to get gt solution
    gt_img = fastmri.ifft2c(kspace)
    
    compressed_imgs = []
    

    #Compress to 8 virtual coils and crop
    compressed_img = ImageCropandKspaceCompression(tensor_to_complex_np(gt_img).transpose(1,2,0),
                                                   img_size, num_vcoils, vh)

        
    #Get the kspace for compressed imgs
    compressed_k = fastmri.fft2c(to_tensor(compressed_img).permute(2,0,1,3))
    
    
    return compressed_img, compressed_k


class MulticoilTransform:
    
    def __init__(self, mask_type = None, img_size=320, accel_rate = 4, num_vcoils=8):

        self.mask_type = mask_type
        self.img_size = img_size
        self.accel_rate = accel_rate
        self.num_vcoils = num_vcoils
        
    def __call__(self, kspace, max_val, vh):
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

        #kspace is dimension [num_coils, size0, size1, 2]
        kspace = to_tensor(kspace)
        
        #Compress to virtual coils
        gt_img, gt_k = get_compressed(kspace, self.img_size, num_vcoils = self.num_vcoils, vh = vh)

        # Stack the coils and real and imaginary
        gt_img = to_tensor(gt_img).permute(2,3,0,1).reshape(-1,self.img_size,self.img_size).unsqueeze(0)

        #Apply the mask
        mask = get_mask(accel=self.accel_rate, size=self.img_size, mask_type=self.mask_type)
        masked_kspace = apply_mask(gt_k, mask)
        
        #Get the zf imgs
        masked_img = fastmri.ifft2c(masked_kspace).permute(0,3,1,2).reshape(-1,self.img_size,self.img_size).unsqueeze(0)

        #Normalized based on the 95th percentile max value of the magnitude
        zf_img = masked_img/max_val
        gt_img = gt_img/max_val
        

        return zf_img, gt_img, mask
    


class MulticoilDataset(torch.utils.data.Dataset):
    def __init__(self, root, max_val_dir, img_size=320, mask_type = 's4', accel_rate = 4, scan_type = None, num_vcoils = 8, slice_range=None,  **kwargs):
        ''' 
        scan_type: None, 'CORPD_FBK', 'CORPDFS_FBK' for knee
        scan_type: None, 'AXT2'
        '''

        self.root = root
        self.img_size = img_size
        self.mask_type = mask_type
        self.accel_rate = accel_rate
        self.max_val_dir = max_val_dir
        self.examples = []
        
        self.multicoil_transf = MulticoilTransform(mask_type=self.mask_type,
                                                img_size = self.img_size,
                                                accel_rate = 4,
                                                num_vcoils = num_vcoils,
                                                )
        
        self.slice_range = slice_range
        

        files = list(Path(root).iterdir())

        print('Loading Data')
        for fname in tqdm(sorted(files)):
            
            #Skip non-data files
            if fname.name[0] == '.':
                continue
            
            #Recover the metadata
            metadata, num_slices = retrieve_metadata(fname)
            
            with h5py.File(fname, "r") as hf:
                
                #Get the attributes of the volume
                attrs = dict(hf.attrs)
                attrs.update(metadata)
                
                
                if scan_type is not None:
                    if attrs["acquisition"] != scan_type or attrs['encoding_size'][1] < img_size or hf['kspace'].shape[1] <= num_vcoils:
                        continue
                

                #Use all the slices if a range is not specified
                if self.slice_range is None:
                    num_slices = hf['kspace'].shape[0]
                    self.slice_range = [0, num_slices]
                
                
                
            self.examples += [(fname, slice_ind) for slice_ind in range(self.slice_range[0],self.slice_range[1])]


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        fname, dataslice = self.examples[i]
        
        with h5py.File(os.path.join(self.max_val_dir, fname.name), 'r') as hf:
            max_val = hf.attrs['max_val']
            vh = hf['vh'][dataslice]
            #vh = None
        
        
        with h5py.File(fname, "r") as hf:
            
            #Get the compressed target kspace
            kspace = hf['kspace'][dataslice]
            
            acquisition = hf.attrs['acquisition']
            
            zf_img, gt_img, mask = self.multicoil_transf(kspace=kspace,  
                                                                 max_val=max_val,
                                                                 vh=vh)
            
            zf_img = zf_img.squeeze(0)
            gt_img = gt_img.squeeze(0)
            
                
        return (
            zf_img.float(), #TODO change back
            gt_img.float(),
            0,
            mask,
            np.float32(max_val),
            acquisition,
            fname.name,
            dataslice,
        )
    
    
class FastMRIDataModule(pl.LightningDataModule):

    def __init__(self, base_path, batch_size:int = 32, num_data_loader_workers:int = 4, **kwargs):
        """
        Initialize the data module for the LoDoPaB-CT dataset.

        Parameters
        ----------
        batch_size : int, optional
            Size of a mini batch.
            The default is 4.
        num_data_loader_workers : int, optional
            Number of workers.
            The default is 8.

        Returns
        -------
        None.

        """
        super().__init__()

        self.batch_size = batch_size
        self.num_data_loader_workers = num_data_loader_workers
        #self.data_range = [-6, 6]
        
        self.base_path = base_path
        self.center_frac = kwargs['center_frac']
        self.accel_rate = kwargs['accel_rate']
        self.img_size = kwargs['img_size']
        self.use_complex = kwargs['complex']
        self.challenge = kwargs['challenge']
        self.vol = kwargs['vol']
        #self.mri_type = kwargs['mri_type'] #'knee', 'brain'
        self.img_type = kwargs['img_type'] #'mag', 'real'
        self.mask_type = kwargs['mask_type']
        
        if 'num_vcoils' in kwargs:
            self.num_vcoils = kwargs['num_vcoils']
        else:
            self.num_vcoils = 8
        
        
        
        if 'scan_type' in kwargs:
            self.scan_type = kwargs['scan_type']
        else:
            self.scan_type = None
            
        if 'unet_dir' in kwargs:
            self.unet_dir = kwargs['unet_dir']
        else:
            self.unet_dir = None
            
        if 'slice_range' in kwargs:
            self.slice_range = kwargs['slice_range']
        else:
            self.slice_range = None
        


    def prepare_data(self):
        """
        Preparation steps like downloading etc. 
        Don't use self here!

        Returns
        -------
        None.

        """
        None

    def setup(self, stage:str = None):
        """
        This is called by every GPU. Self can be used in this context!

        Parameters
        ----------
        stage : str, optional
            Current stage, e.g. 'fit' or 'test'.
            The default is None.

        Returns
        -------
        None.

        """
        train_dir = os.path.join(self.base_path, '{0}_{1}'.format(self.challenge, 'train'))
        val_dir = os.path.join(self.base_path, '{0}_{1}'.format(self.challenge, 'val'))
        test_dir = os.path.join(self.base_path, 'small_T2_test')
        
        max_val_dir_train = os.path.join(self.base_path, '{0}_{1}_{2}coils'.format(self.challenge, 'train', self.num_vcoils))
        max_val_dir_val = os.path.join(self.base_path, '{0}_{1}_{2}coils'.format(self.challenge, 'val', self.num_vcoils))
        max_val_dir_test = os.path.join(self.base_path, 'multicoil_small_T2_test_8coils')


        # Assign train/val datasets for use in dataloaders
        self.train = MulticoilDataset(train_dir, 
                                      max_val_dir_train, 
                                      self.img_size, self.mask_type, 
                                      self.accel_rate, self.scan_type, 
                                      self.num_vcoils,
                                      self.slice_range
                                      )
        self.val = MulticoilDataset(val_dir,
                                    max_val_dir_val, 
                                    self.img_size, self.mask_type, 
                                    self.accel_rate, self.scan_type,
                                    self.num_vcoils,
                                    self.slice_range
                                    )
        
        self.test = MulticoilDataset(test_dir,
                                    max_val_dir_test, 
                                    self.img_size, self.mask_type, 
                                    self.accel_rate, self.scan_type,
                                    self.num_vcoils,
                                    slice_range=[0,6]
                                    )



    def train_dataloader(self):
        """
        Data loader for the training data.

        Returns
        -------
        DataLoader
            Training data loader.

        """
        return DataLoader(self.train, batch_size=self.batch_size,
                          num_workers=self.num_data_loader_workers,
                          shuffle=True, pin_memory=True)

    def val_dataloader(self):
        """
        Data loader for the validation data.

        Returns
        -------
        DataLoader
            Validation data loader.

        """
        return DataLoader(self.val, batch_size=self.batch_size,
                          num_workers=self.num_data_loader_workers,
                          shuffle=False, pin_memory=True)
    
    def test_dataloader(self):
        """
        Data loader for the validation data.

        Returns
        -------
        DataLoader
            Validation data loader.

        """
        return DataLoader(self.test, batch_size=self.batch_size,
                          num_workers=self.num_data_loader_workers,
                          shuffle=False, pin_memory=True)

    
    
if __name__ == '__main__':
    # Location of the dataset
    base_dir = "/storage/fastMRI_brain/data/"
    #base_dir= '/scratch/fastMRI/data/'
    #base_dir = '/storage/data/'
    #base_dir = '../../datasets/fastMRI/data'

        
    
    #%%
    
    
    kwargs = {'center_frac': 0.0807,
              'accel_rate': 4,
              'img_size': 384,
              'complex': True,
              'challenge': 'multicoil',
              'vol': True,
              'img_type': 'mag',
              'mask_type': 's4a5',
              'scan_type': 'AXT2',
              'num_vcoils': 8,
              'slice_range': [0,8]
              }
    
    data = FastMRIDataModule(base_dir, batch = 16, **kwargs)
    data.prepare_data()
    data.setup()
    #dataset = MulticoilDataset(dataset_dir, max_val_dir, img_size, mask_type)
    #dataset[0][0]
    
    #%%
    
    # loader = data.val_dataloader()
    
    # start = time.time()
    # for i,batch in enumerate(tqdm(loader)):
    #     continue
    # end = time.time()
    # print(end-start)
    
    #%%
    zf_img = data.train[0][0]
    gt_img = data.train[0][1]
    norm_val = torch.tensor(data.val[0][4])
    
    zf_img = network_utils.unnorm(zf_img.unsqueeze(0), norm_val)
    gt_img = network_utils.unnorm(gt_img.unsqueeze(0), norm_val)
    maps = network_utils.get_maps(zf_img, 31)
    
    viz.show_multicoil_combo(gt_img, maps)
    viz.show_multicoil_combo(zf_img, maps)
        
        
    # show_multicoil_combo(data.train[25][0], data.train[25][2])
    
    # loader = data.train_dataloader()
    # x = next(iter(loader))
    # show_multicoil_combo(x[1].cpu(), x[2].cpu())
    
    '''
    #Get a list of all the h5py files
    files = list(Path(dataset_dir).iterdir())
    
    #For each file, perform the transformation and save to a new file
    for fname in sorted(files):
        
        #Recover the metadata
        metadata, num_slices = retrieve_metadata(fname)
        
        with h5py.File(fname, 'r') as hf:
            
            for dataslice in range(15,16):
                
                #Get the kspace
                kspace = hf['kspace'][dataslice]
                
                break
            break
    
#Get the images from the kspace
full_imgs = tensor_to_complex_np(fastmri.ifft2c(to_tensor(kspace)))
viz.show_img(to_tensor(full_imgs).permute(0,3,1,2), val_range = (0, 1.97e-5))

compressed_img = ImageCropandKspaceCompression(full_imgs.transpose(1,2,0), img_size)
viz.show_img(to_tensor(compressed_img).permute(2,3,0,1), val_range = (0, 1.97e-5))

#Get the kspace for the compressed imgs
compressed_k = fastmri.fft2c(to_tensor(compressed_img).permute(2,0,1,3))

# y is multicoil undersampled measurements - needs to be complex np datatype
# calib_width is ACS size
full_maps = mr.app.EspiritCalib(tensor_to_complex_np(compressed_k), 
                           calib_width=32,
                           show_pbar=False, 
                           crop=0.70, 
                           kernel_width=6).run()

#View the sensitivity maps
viz.show_img(to_tensor(full_maps).permute(0,3,1,2))

#Show SENSE estimate
S = sp.linop.Multiply((img_size,img_size), full_maps)
combo_img = S.H * compressed_img.transpose(2,0,1)
viz.show_img(to_tensor(combo_img).permute(2,0,1),val_range = (0, 9.97e-5))


#%% Apply to masked kspace
mask = get_mask(accel=accel_rate, size=img_size, mask_type=mask_type)
masked_kspace = apply_mask(compressed_k, mask)

masked_imgs = fastmri.ifft2c(masked_kspace)
viz.show_img(masked_imgs.permute(0,3,1,2), val_range = (0, 3.97e-5))

# y is multicoil undersampled measurements - needs to be complex np datatype
# calib_width is ACS size
sense_maps = mr.app.EspiritCalib(tensor_to_complex_np(masked_kspace), 
                           calib_width=13,
                           show_pbar=False, 
                           crop=0.70, 
                           kernel_width=6).run()

#View the sensitivity maps
viz.show_img(to_tensor(sense_maps).permute(0,3,1,2))

#Show SENSE estimate
S = sp.linop.Multiply((img_size,img_size), sense_maps)
combo_img = S.H * tensor_to_complex_np(masked_imgs)
viz.show_img(to_tensor(combo_img).permute(2,0,1),val_range = (0, 9.97e-5))
'''
