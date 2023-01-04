#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 14:21:11 2022

@author: jeff
"""

import torch
from pathlib import Path
import h5py
import os
import time
import fastmri
import pytorch_lightning as pl
from torch.utils.data import DataLoader

import sys

sys.path.append("..")
from util import viz

from fastmri.data import transforms as mri_transforms


class ConditionalSliceDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, use_complex = False):

        self.root = root
        self.transforms = transforms
        self.use_complex = use_complex
        self.examples = []

        files = list(Path(root).iterdir())

        for fname in sorted(files):

            with h5py.File(fname, "r") as hf:

                num_slices = hf["zero_filled_imgs"].shape[0]

                self.examples += [(fname, slice_ind) for slice_ind in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        fname, dataslice = self.examples[i]

        with h5py.File(fname, "r") as hf:
            #print(hf.keys())
            zf_img = mri_transforms.to_tensor(hf["zero_filled_imgs"][dataslice])
            target_img = mri_transforms.to_tensor(hf["target_imgs"][dataslice])
            mask = mri_transforms.to_tensor(hf["mask"][dataslice])
            mean = hf["means"][dataslice]
            std = hf["stds"][dataslice]
            acquisition = hf.attrs["acquisition"]
            fname = hf.attrs["fname"]
            slice_num = hf["slice_nums"][dataslice]
            
        if not self.use_complex:
            zf_img = zf_img.unsqueeze(0)
            target_img = target_img.unsqueeze(0)
            
        return (
            zf_img,
            target_img,
            mask,
            mean,
            std,
            acquisition,
            fname,
            slice_num,
        )
        
    
    
class ConditionalSliceDatasetVol(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, use_complex = False, scan_type = None, img_type = 'mag'):
        ''' 
        scan_type: None, 'CORPD_FBK', 'CORPDFS_FBK'
        '''

        self.root = root
        self.transforms = transforms
        self.use_complex = use_complex
        self.img_type = img_type
        self.examples = []

        files = list(Path(root).iterdir())

        for fname in sorted(files):

            with h5py.File(fname, "r") as hf:
                
                if scan_type is not None:
                    if hf.attrs['acquisition'] != scan_type:
                        continue

                num_slices = hf["zero_filled_imgs"].shape[0]

                self.examples += [(fname, slice_ind) for slice_ind in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        fname, dataslice = self.examples[i]

        with h5py.File(fname, "r") as hf:
            #print(hf.keys())
            zf_img = mri_transforms.to_tensor(hf["zero_filled_imgs"][dataslice])
            target_img = mri_transforms.to_tensor(hf["target_imgs"][dataslice])
            mask = mri_transforms.to_tensor(hf["mask"][0])
            normstd = hf.attrs['normstd']
            normstdval = hf.attrs['normstdval']
            acquisition = hf.attrs["acquisition"]
            fname = hf.attrs["fname"]
            slice_num = hf["slice_nums"][dataslice]
            
        if self.img_type == 'real':
            zf_img = zf_img[0].unsqueeze(0)
            target_img = target_img[0].unsqueeze(0)
        
        if not self.use_complex:
            zf_img = zf_img.unsqueeze(0)
            target_img = target_img.unsqueeze(0)
            
        return (
            zf_img, #TODO change back
            target_img,
            mask,
            normstd,
            normstdval,
            acquisition,
            fname,
            slice_num,
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
        
        if 'scan_type' in kwargs:
            self.scan_type = kwargs['scan_type']
        else:
            self.scan_type = None
        
        if 'patches' in kwargs:
            self.patches = kwargs['patches']
        else:
            self.patches = False
            
        if 'mask_type' in kwargs:
            self.mask_type = kwargs['mask_type']
        else:
            self.mask_type = None

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
        
        if self.vol:
            
            if self.mask_type is not None:
                
                # Location of the new dataset
                if self.use_complex:
                    
                    dataset_dir = os.path.join(
                        self.base_path,
                        "{0}_{1}_size{2}_accel{3}_complex_vol_gro{4}".format(
                            self.challenge, "train", self.img_size, self.accel_rate, self.mask_type
                        ),
                    )
                    valset_dir = os.path.join(
                        self.base_path,
                        "{0}_{1}_size{2}_accel{3}_complex_vol_gro{4}".format(
                            self.challenge, "val", self.img_size, self.accel_rate, self.mask_type
                        ),
                    )
                else:

                    dataset_dir = os.path.join(
                        self.base_path,
                        "{0}_{1}_size{2}_accel{3}_vol_gro{4}".format(
                            self.challenge, "train", self.img_size, self.accel_rate, self.mask_type
                        ),
                    )
                    valset_dir = os.path.join(
                        self.base_path,
                        "{0}_{1}_size{2}_accel{3}_vol_gro{4}".format(
                            self.challenge, "val", self.img_size, self.accel_rate, self.mask_type
                        ),
                    )
            
            else:
                # Location of the new dataset
                if self.use_complex:
                    
                    if self.patches:
                        dataset_dir = os.path.join(
                            self.base_path,
                            "{0}_{1}_size{2}_centerfrac{3}_accel{4}_complex_vol_patches".format(
                                self.challenge, "train", self.img_size, self.center_frac, self.accel_rate
                            ),
                        )
                        valset_dir = os.path.join(
                            self.base_path,
                            "{0}_{1}_size{2}_centerfrac{3}_accel{4}_complex_vol_patches".format(
                                self.challenge, "val", self.img_size, self.center_frac, self.accel_rate
                            ),
                        )
                    else:
                        dataset_dir = os.path.join(
                            self.base_path,
                            "{0}_{1}_size{2}_centerfrac{3}_accel{4}_complex_vol".format(
                                self.challenge, "train", self.img_size, self.center_frac, self.accel_rate
                            ),
                        )
                        valset_dir = os.path.join(
                            self.base_path,
                            "{0}_{1}_size{2}_centerfrac{3}_accel{4}_complex_vol".format(
                                self.challenge, "val", self.img_size, self.center_frac, self.accel_rate
                            ),
                        )
                else:
                    if self.patches:
                        dataset_dir = os.path.join(
                            self.base_path,
                            "{0}_{1}_size{2}_centerfrac{3}_accel{4}_vol_patches".format(
                                self.challenge, "train", self.img_size, self.center_frac, self.accel_rate
                            ),
                        )
                        valset_dir = os.path.join(
                            self.base_path,
                            "{0}_{1}_size{2}_centerfrac{3}_accel{4}_vol_patches".format(
                                self.challenge, "val", self.img_size, self.center_frac, self.accel_rate
                            ),
                        )
                    else:
                        
                        dataset_dir = os.path.join(
                            self.base_path,
                            "{0}_{1}_size{2}_centerfrac{3}_accel{4}_vol".format(
                                self.challenge, "train", self.img_size, self.center_frac, self.accel_rate
                            ),
                        )
                        valset_dir = os.path.join(
                            self.base_path,
                            "{0}_{1}_size{2}_centerfrac{3}_accel{4}_vol".format(
                                self.challenge, "val", self.img_size, self.center_frac, self.accel_rate
                            ),
                        )
    
            # Assign train/val datasets for use in dataloaders
            self.train = ConditionalSliceDatasetVol(dataset_dir,use_complex = self.use_complex, scan_type = self.scan_type, img_type = self.img_type)
            self.val = ConditionalSliceDatasetVol(valset_dir,use_complex = self.use_complex, scan_type = self.scan_type, img_type = self.img_type)
            
        else:
            # Location of the new dataset
            if self.use_complex:
                dataset_dir = os.path.join(
                    self.base_path,
                    "{0}_{1}_size{2}_centerfrac{3}_accel{4}_complex".format(
                        self.challenge, "train", self.img_size, self.center_frac, self.accel_rate
                    ),
                )
                valset_dir = os.path.join(
                    self.base_path,
                    "{0}_{1}_size{2}_centerfrac{3}_accel{4}_complex".format(
                        self.challenge, "val", self.img_size, self.center_frac, self.accel_rate
                    ),
                )
            else:
                dataset_dir = os.path.join(
                    self.base_path,
                    "{0}_{1}_size{2}_centerfrac{3}_accel{4}".format(
                        self.challenge, "train", self.img_size, self.center_frac, self.accel_rate
                    ),
                )
                valset_dir = os.path.join(
                    self.base_path,
                    "{0}_{1}_size{2}_centerfrac{3}_accel{4}".format(
                        self.challenge, "val", self.img_size, self.center_frac, self.accel_rate
                    ),
                )
    
            # Assign train/val datasets for use in dataloaders
            self.train = ConditionalSliceDataset(dataset_dir,use_complex = self.use_complex)
            self.val = ConditionalSliceDataset(valset_dir,use_complex = self.use_complex)


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


#%%
# Location of the dataset
base_dir = "../../datasets/fastMRI/data/"
dataset_dir = os.path.join(base_dir, "singlecoil_train")

# Parameters
center_frac = 0.08
accel_rate = 4
img_size = 320
challenge = "singlecoil"
dataset_type = "train"
use_complex = True

# Location of the new dataset
if not use_complex:
    new_dir = os.path.join(
        base_dir,
        "{0}_{1}_size{2}_centerfrac{3}_accel{4}".format(
            challenge, dataset_type, img_size, center_frac, accel_rate
        ),
    )
else:
    new_dir = os.path.join(
        base_dir,
        "{0}_{1}_size{2}_centerfrac{3}_accel{4}_complex".format(
            challenge, dataset_type, img_size, center_frac, accel_rate
        ),
    )

if __name__ == "__main__":

    dataset = ConditionalSliceDataset(new_dir, use_complex=use_complex)

    start = time.time()
    # for i, data in enumerate(trainloader):
    for i in range(len(dataset)):
        test = dataset[i]

    end = time.time()
    print(end - start)

    if not use_complex:
        viz.show_tensor_imgs(dataset[0][1])
    else:
        sample = dataset[25]
        viz.show_tensor_imgs(fastmri.complex_abs(sample[1].permute(1,2,0)))
        
        mean = sample[3]
        std = sample[4]

        # Show that this is correct
        zf_kspace = fastmri.fft2c((sample[0]*(std + 1e-11)+mean).permute(1,2,0))
        gt_kspace = fastmri.fft2c((sample[1]*(std + 1e-11)+mean).permute(1,2,0))

        viz.plot_kspace(zf_kspace)
        viz.plot_kspace(gt_kspace)
