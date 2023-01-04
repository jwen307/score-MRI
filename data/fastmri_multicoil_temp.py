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


class MulticoilTransform:
    
    def __init__(self, mask_type = None, img_size=320, accel_rate = 4):

        self.mask_type = mask_type
        self.img_size = img_size
        self.accel_rate = accel_rate
        
    def __call__(self, kspace, max_val):
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

        #kspace is dimension [num_slices, num_coils, size0, size1, 2]
        kspace = to_tensor(kspace)

        # inverse Fourier transform to get gt solution
        gt_img = fastmri.ifft2c(kspace).permute(0,3,1,2).reshape(-1,self.img_size,self.img_size).unsqueeze(0)

        #Apply the mask
        mask = get_mask(accel=self.accel_rate, size=self.img_size, mask_type=self.mask_type)
        masked_kspace = apply_mask(kspace, mask)
        
        #Get the zf imgs
        masked_img = fastmri.ifft2c(masked_kspace).permute(0,3,1,2).reshape(-1,self.img_size,self.img_size).unsqueeze(0)

        #Normalized based on the 95th percentile max value of the magnitude
        zf_img = masked_img/max_val
        gt_img = gt_img/max_val
        

        return zf_img, gt_img, mask


class MulticoilDataset(torch.utils.data.Dataset):
    def __init__(self, root, max_val_dir, img_size=320, mask_type = 's4', accel_rate = 4, scan_type = None, **kwargs):
        ''' 
        scan_type: None, 'CORPD_FBK', 'CORPDFS_FBK'
        '''

        self.root = root
        self.img_size = img_size
        self.mask_type = mask_type
        self.accel_rate = accel_rate
        self.max_val_dir = max_val_dir
        self.examples = []
        
        self.multicoil_transf = MulticoilTransform(mask_type=self.mask_type,
                                                img_size = self.img_size
                                                )
        
        if 'unet_dir' in kwargs:
            self.unet_dir = kwargs['unet_dir']
        else:
            self.unet_dir = None

        files = list(Path(root).iterdir())

        print('Loading Data')
        for fname in tqdm(sorted(files)):
            
            with h5py.File(fname, "r") as hf:
                
                if scan_type is not None:
                    if hf.attrs["acquisition"] != scan_type:
                        continue
                
                num_slices = hf['target_kspace'].shape[0]
                
                
            self.examples += [(fname, slice_ind) for slice_ind in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        fname, dataslice = self.examples[i]
        
        # with h5py.File(os.path.join(self.max_val_dir, fname.name), 'r') as hf:
        #     max_val = hf.attrs['max_val']
        #     maps = hf['sense_maps'][dataslice]
        max_val = 1
        maps=0
        
        with h5py.File(fname, "r") as hf:
            
            #Get the compressed target kspace
            kspace = hf['target_kspace'][dataslice]
            
            mask_type = hf.attrs['mask_type']
            acquisition = hf.attrs['acquisition']
            
            zf_img, gt_img, mask = self.multicoil_transf(kspace=kspace,  
                                                                 max_val=max_val)
            
            zf_img = zf_img.squeeze(0)
            gt_img = gt_img.squeeze(0)
            
        #Give the UNet predictions if requested
        if self.unet_dir is not None:
            unet_fname = os.path.join(self.unet_dir, fname.name)
            
            #UNet prediction is already normalized
            with h5py.File(unet_fname, "r") as hf:
                unet_pred = hf['unet_recons'][dataslice]
                unet_pred = torch.tensor(unet_pred).permute(0,3,1,2).reshape(-1,self.img_size,self.img_size)
                
                
            return (
                unet_pred.float(), #TODO change back
                gt_img,
                maps,
                mask,
                np.float32(max_val),
                acquisition,
                fname.name,
                dataslice,
            )
                
            
            
        return (
            zf_img.float(), #TODO change back
            gt_img.float(),
            maps,
            mask,
            np.float32(max_val),
            acquisition,
            fname.name,
            dataslice,
        )
    
    
class FastMRIDataModule(pl.LightningDataModule):

    def __init__(self, base_path, batch_size:int = 32, num_data_loader_workers:int = 20, **kwargs):
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
        train_dir = os.path.join(self.base_path, '{0}_{1}_{2}coils'.format(self.challenge, 'train', self.num_vcoils))
        val_dir = os.path.join(self.base_path, '{0}_{1}_{2}coils'.format(self.challenge, 'val', self.num_vcoils))
        
        max_val_dir_train = os.path.join(self.base_path,'{0}_maxval_{2}coils_{1}'.format('train', self.mask_type,self.num_vcoils))
        max_val_dir_val = os.path.join(self.base_path,'{0}_maxval_{2}coils_{1}'.format('val', self.mask_type, self.num_vcoils))



        if self.unet_dir is not None:
            unet_dir_train = os.path.join(self.unet_dir, 'outputs/train')
            unet_dir_val = os.path.join(self.unet_dir, 'outputs/val')

            # Assign train/val datasets for use in dataloaders
            self.train = MulticoilDataset(train_dir, 
                                          max_val_dir_train, 
                                          self.img_size, self.mask_type, 
                                          self.accel_rate, self.scan_type,
                                          unet_dir = unet_dir_train
                                          )
            self.val = MulticoilDataset(val_dir,
                                        max_val_dir_val, 
                                        self.img_size, self.mask_type, 
                                        self.accel_rate, self.scan_type,
                                        unet_dir = unet_dir_val
                                        )
        
        else:
            # Assign train/val datasets for use in dataloaders
            # self.train = MulticoilDataset(train_dir, 
            #                               max_val_dir_train, 
            #                               self.img_size, self.mask_type, 
            #                               self.accel_rate, self.scan_type,
            #                               )
            self.val = MulticoilDataset(val_dir,
                                        max_val_dir_val, 
                                        self.img_size, self.mask_type, 
                                        self.accel_rate, self.scan_type,
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

#Convert image so there are two channels for real and imaginary and c coils
def chans_to_coils(multicoil_imgs):
    
    #Add a batch dimension if needed
    if len(multicoil_imgs.shape) < 4:
        multicoil_imgs = multicoil_imgs.unsqueeze(0)
        
    b, c, h, w = multicoil_imgs.shape
        
    #Split into real and imag
    #multicoil_imgs = multicoil_imgs.reshape(b, -1, 2, h, w)
    multicoil_imgs = torch.stack([multicoil_imgs[:,0:int(c):2,:,:], multicoil_imgs[:,1:int(c):2,:,:]], dim=-1)
    #multicoil_imgs = multicoil_imgs.permute(0,1,3,4,2).contiguous()
    
    return multicoil_imgs
    

def show_multicoil_combo(multicoil_imgs, maps):
    
    #Get the coil images from the channels
    multicoil_imgs = chans_to_coils(multicoil_imgs)
    
    b, num_coils, h, w, _ = multicoil_imgs.shape
    
    if torch.is_tensor(maps):
        if len(maps.shape) <4:
            maps = maps.unsqueeze(0)
        maps = maps.numpy()
    else:
        if len(maps.shape)<4:
            maps = np.expand_dims(maps, axis=0)
        
    combo_imgs = []
    for i in range(b):
        with sp.Device(0):
            #Show SENSE estimate
            S = sp.linop.Multiply((h,w), maps[i])
            combo_img = S.H * tensor_to_complex_np(multicoil_imgs[i])
            
            combo_imgs.append(to_tensor(combo_img))
        
    combo_imgs = torch.stack(combo_imgs)
    #viz.show_img(combo_imgs, val_range = (0, 5))
    viz.show_img(combo_imgs)
    
    
if __name__ == '__main__':
    # Location of the dataset
    base_dir = "/storage/fastMRI/data/"
    #base_dir= '/scratch/fastMRI/data/'
    #base_dir = '/storage/data/'
    #base_dir = '../../datasets/fastMRI/data'
    
    
    dataset_type = 'train'
    challenge = 'multicoil'
    dataset_dir = os.path.join(base_dir, '{0}_{1}_8coils'.format(challenge, dataset_type))
    
    #Parameters
    accel_rate = 4
    img_size = 320
    
    use_complex = True
    vol = True #Normalize across the volume instead of individual images
    normstd = 'normalize' #Either 'normalize' or 'standardize'
    mask_type = 's4'
    
    max_val_dir = os.path.join(base_dir, '{0}_maxval_8coils_{1}'.format(dataset_type,mask_type))
    

    
    #%%
    
    
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
    
    data = FastMRIDataModule(base_dir, batch = 16, **kwargs)
    data.prepare_data()
    data.setup()
    #dataset = MulticoilDataset(dataset_dir, max_val_dir, img_size, mask_type)
    #dataset[0][0]
    
    #%% Get images for the score-based model
    img_idx = 100
    
    img = data.val[img_idx][1]
    mask = data.val[img_idx][3]
    
    #Convert to a complex np array
    img = network_utils.chans_to_coils(img)
    img = fastmri.tensor_to_complex_np(img)
    
    #Get a 2D mask
    mask2D = mask.permute(0,2,1).repeat(1,mask.shape[1],1).numpy()
    
    #Save the arrays
    np.save('val{0}.npy'.format(img_idx), img)
    np.save('val{0}_mask.npy'.format(img_idx),mask2D)
    
    
    
    
    
    
    #%%
    
        
        
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
