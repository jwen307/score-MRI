#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 11:50:00 2022

@author: wen.254
"""

import numpy as np
import scipy.io 
import torch


def get_mask(accel=4, size=320, mask_type='s4'):
    
    #Load the matrix
    mask = scipy.io.loadmat('data/masks/mask_accel{0}_size{1}_gro_{2}.mat'.format(accel, size, mask_type))
    mask = mask['samp'].astype('float32')
    mask = torch.from_numpy(mask).unsqueeze(0)
    
    
    return mask

def apply_mask(data, mask):
    
    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data


#mask_type='s4'
#mask = scipy.io.loadmat('mask_accel4_size320_gro_{0}.mat'.format(mask_type))