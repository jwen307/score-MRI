#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 06:56:12 2022

@author: jeff
"""


import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers,seed_everything
import os
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from typing import Dict, Optional, Sequence, Tuple, Union
import time

import sigpy as sp
import sigpy.mri as mr

import sys
sys.path.append("..")

import fastmri
from fastmri.data.transforms import to_tensor, tensor_to_complex_np
from datasets.fastmri_multicoil import FastMRIDataModule
from util import util, viz, network_utils, mail


# TODO Change these for new models
from models.pl_models.unet_multicoil import UNetMulticoil
from models.net_builds import build_unet
from train.configs.config_unet_multicoil import Config


# Location of the dataset
base_dir = "/storage/fastMRI/data/"


#%% Network Setup
# UNet checkpoint information
load_ckpt_dir = '/storage/jeff/mri_cinn1/logs/lightning_logs/version_11'
load_ckpt_epoch = 49
load_ckpt_step = 108050


# Get the ckpt for the preprocessing UNet
ckpt = os.path.join(load_ckpt_dir, 
                    'checkpoints', 
                    'epoch={0}-step={1}.ckpt'.format(load_ckpt_epoch, load_ckpt_step))

#Get the configuration file
config_file = os.path.join(load_ckpt_dir, 'config.pkl')
net_args = util.read_pickle(config_file)

#Setup the network
if 'num_vcoils' in net_args['train_args']:
    in_ch = net_args['train_args']['num_vcoils'] * 2
else:
    in_ch = 16

#Set the input dimensions
img_size = net_args['train_args']['img_size']
input_dims = [in_ch, img_size, img_size]

#Pick which build functions to use
builds = [
            build_unet.build0,
          ]
build_num = net_args['train_args']['build_num']

seed_everything(1, workers=True)

print("Loading previous checkpoint")
#model.load_state_dict(ckpt['state_dict'])
model = UNetMulticoil.load_from_checkpoint(ckpt, strict=False, input_dims= input_dims, 
                                        build_func = builds[build_num], 
                                        net_args=net_args )

model.eval()
model.cuda()



#%% Data setup
#Get the data
data = FastMRIDataModule(base_dir, 
                         num_data_loader_workers=0,
                         **net_args['train_args'],
                         )
data.prepare_data()
data.setup()


#%% Run the train loader and store output of UNet in dictionary
#Dictionary for the reconstructions
reconstructions = defaultdict(list)

print('Getting the reconstructions')
with torch.no_grad():
    for i, batch in enumerate(tqdm(data.train_dataloader())):
        obs, gt, maps, mask, std, acquisition, fname, slice_num = batch
        
        
        obs = obs.to(model.device)
        mask = mask.to(model.device)
        std = std.to(model.device)
        
        # Get the reconstructions
        samples = model.reconstruct(obs, maps, mask, std, multicoil=True)
        # Normalize the samples
        reco = samples / std.reshape(-1,1,1,1,1).cpu()
        reco = reco.cpu()
        
        # Reconstructiosn saved as (batch_size, num_coils, img_size, img_size, 2)
        for k in range(gt.shape[0]):
            reconstructions[fname[k]].append((int(slice_num[k]), reco[k]))
    
#%% 
# Save the reconstructions
save_dir = os.path.join(load_ckpt_dir, 'outputs', 'train')
Path(save_dir).mkdir(parents=True, exist_ok=True)
print('Saving Reconstructions')

for fname in tqdm(reconstructions):

    #Organize the slices to be in order
    reconstructions[fname] = np.stack([out for _, out in sorted(reconstructions[fname])])


    with h5py.File(os.path.join(save_dir, fname), 'w') as nf:
        recon_nf = nf.create_dataset('unet_recons', reconstructions[fname].shape, data=reconstructions[fname])


#%% Run the val loader and store output of UNet in dictionary
#Dictionary for the reconstructions
reconstructions = defaultdict(list)

print('Getting the reconstructions')
with torch.no_grad():
    for i, batch in enumerate(tqdm(data.val_dataloader())):
        obs, gt, maps, mask, std, acquisition, fname, slice_num = batch
        
        
        obs = obs.to(model.device)
        mask = mask.to(model.device)
        std = std.to(model.device)
        
        # Get the reconstructions
        samples = model.reconstruct(obs, maps, mask, std, multicoil=True)
        # Normalize the samples
        reco = samples / std.reshape(-1,1,1,1,1).cpu()
        reco = reco.cpu()
        
        for k in range(gt.shape[0]):
            reconstructions[fname[k]].append((int(slice_num[k]), reco[k]))
    
#%% 
# Save the reconstructions
save_dir = os.path.join(load_ckpt_dir, 'outputs', 'val')
Path(save_dir).mkdir(parents=True, exist_ok=True)
print('Saving Reconstructions')

for fname in tqdm(reconstructions):

    #Organize the slices to be in order
    reconstructions[fname] = np.stack([out for _, out in sorted(reconstructions[fname])])


    with h5py.File(os.path.join(save_dir, fname), 'w') as nf:
        recon_nf = nf.create_dataset('unet_recons', reconstructions[fname].shape, data=reconstructions[fname])







    