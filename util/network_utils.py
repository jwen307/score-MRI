#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 09:18:44 2022

@author: jeff
"""
import torch
import numpy as np
import fastmri
from fastmri.data.transforms import to_tensor, tensor_to_complex_np

import sigpy as sp
import sigpy.mri as mr

#Function to give pixel values between 0-255
def quantize(x):
    #Assumes x is between 0 and 1
    return (x*255).to(torch.int32)


def print_num_params(model):
    num_params = sum([np.prod(p.shape) for p in model.parameters()])
    print("Number of parameters: {:,}".format(num_params))
    
#Get the preimage dataset and dataloader
def get_preimgset(model, loader, conditional = False, batch_size = 64, shuffle=True, num_workers=20):
    preimg_list = []
    condition_list = []
    mask_list = []

    std_list = []
    for i, data in enumerate(loader):
        cond = data[0]
        imgs = data[1]
        mask = data[2]
        std = data[4]
        
        #imgs, cond = data
        
        #Get the preimages
        with torch.no_grad():
            if conditional:
                preimgs = model.get_preimages(imgs.cuda(),cond.cuda())
            else:
                preimgs = model.get_preimages(imgs.cuda())
        preimg_list.append(preimgs)
        condition_list.append(cond)
        mask_list.append(mask)
        std_list.append(std)
    
    #Create the preimage set and loader
    pimg_t = torch.cat(preimg_list, dim=0).cpu()
    cond_t = torch.cat(condition_list,dim=0).cpu()
    mask_t = torch.cat(mask_list, dim=0).cpu()
    std_t = torch.cat(std_list, dim=0).cpu()
    
    pimgset = torch.utils.data.TensorDataset(cond_t, pimg_t, mask_t, std_t)

    pimgloader = torch.utils.data.DataLoader(pimgset, batch_size = batch_size, shuffle=shuffle, num_workers = num_workers)

    
    return pimgset, pimgloader



#Get the data consistent reconstruction (checked and it should be good)
def get_dc_image(pred, cond, mask, mean = 0, std=1):
    
    if not torch.is_tensor(std):
        std = torch.tensor(std)
        
    std = std.to(cond.device)
    
    #Got from 16 channels to 8 coils and 2 complex dimensions
    pred = chans_to_coils(pred)
    cond = chans_to_coils(cond)
    
    if len(mask.shape)<4:
        mask = mask.unsqueeze(0)
    
    #Get the zero-filled kspace
    zfkspace = fastmri.fft2c(cond * std.reshape(-1,1,1,1,1) + mean)
    
    #Get the kspace of the predicted image
    pred_kspace = fastmri.fft2c(pred*std.reshape(-1,1,1,1,1) + mean)

    mask2D = mask[0].permute(0,2,1).repeat(1,mask.shape[2],1)
    mask2Dinv = mask2D*-1 + 1
    masked_pred_k = mask2Dinv.unsqueeze(-1).unsqueeze(0).repeat(1,1,1,1,2) * pred_kspace
    
    #Need this since coil compression doesn't make zf kspace exactly zero where mask is
    masked_zfkspace = mask2D.unsqueeze(-1).repeat(1,1,1,2) * zfkspace
    
    dc_pred_k = masked_pred_k + masked_zfkspace
    
    #normalized
    dc_pred = fastmri.ifft2c(dc_pred_k)
    dc_pred = dc_pred / std.reshape(-1,1,1,1,1)
    
    #dc_pred is normalized again
    return dc_pred, dc_pred_k

#Get the data consistent reconstruction (checked and it should be good)
def get_dc_image_singlecoil(pred, cond, mask, mean = 0, std=1):
    
    if not torch.is_tensor(std):
        std = torch.tensor(std)
        
    std = std.to(cond.device)
    
    if len(mask.shape)<4:
        mask = mask.unsqueeze(0)
    
    #Get the zero-filled kspace
    zfkspace = fastmri.fft2c(cond * std.reshape(-1,1,1,1) + mean)
    
    #Get the kspace of the predicted image
    pred_kspace = fastmri.fft2c(pred*std.reshape(-1,1,1,1) + mean)

    mask2D = mask[0].permute(0,2,1).repeat(1,mask.shape[2],1)
    mask2Dinv = mask2D*-1 + 1
    masked_pred_k = mask2Dinv.unsqueeze(-1).repeat(1,1,1,2) * pred_kspace
    
    #Need this since coil compression doesn't make zf kspace exactly zero where mask is
    masked_zfkspace = mask2D.unsqueeze(-1).repeat(1,1,1,2) * zfkspace
    
    dc_pred_k = masked_pred_k + masked_zfkspace
    
    #Not normalized
    dc_pred = fastmri.ifft2c(dc_pred_k)
    dc_pred = dc_pred / std.reshape(-1,1,1,1)
    
    #dc_pred is normalized again
    return dc_pred, dc_pred_k

def apply_mask(x, mask, mean = 0, std=1, inv_mask = False):
    
    if not torch.is_tensor(std):
        std = torch.tensor(std)
        
    std = std.to(x.device)
    
    #Make the mask 2D
    if len(mask.shape)<4:
        mask = mask.unsqueeze(0)
    mask2D = mask[0].permute(0,2,1).repeat(1,mask.shape[2],1)
    if inv_mask:
        mask2D = mask2D*-1 + 1
    
    # Check if we are including the coils or its coil combined
    if x.shape[1] > 2:
        #This is using multiple coils 
        # (batch, num_coils*2, img_size, img_size) -> (batch, num_coils, img_size, img_size, 2)
        x = chans_to_coils(x)
        
        #Get the kspace
        kspace = fastmri.fft2c(x * std.reshape(-1,1,1,1,1) + mean)
        
        #Apply the mask
        masked_k = mask2D.unsqueeze(-1).unsqueeze(0).repeat(1,1,1,1,2) * kspace
        
        #Not normalized
        masked_img = fastmri.ifft2c(masked_k)
        #Renormalized
        masked_img = masked_img / std.reshape(-1,1,1,1,1)
        
        #Get back into (batch, num_coils*2, img_size, img_size)
        masked_img = coils_to_chans(masked_img)
        return masked_img
        
    # This is the coil combined case
    else:
        
        if x.shape[-1] != 2:
            x = x.permute(0,2,3,1)
        
        #Get the kspace
        kspace = fastmri.fft2c(x * std.reshape(-1,1,1,1) + mean)
            
        masked_k = mask2D.unsqueeze(-1).repeat(1,1,1,2) * kspace

        #Not normalized
        masked_img = fastmri.ifft2c(masked_k)
        #Renormalized
        masked_img = masked_img / std.reshape(-1,1,1,1)
    
        #dc_pred is normalized again
        return masked_img.permute(0,3,1,2).float()
    
    
#Convert image so there's 2*num_coil channels
# (batch, num_coils, img_size, img_size, 2) -> (batch, num_coils*2, img_size, img_size)
def coils_to_chans(multicoil_imgs):
    
    #Add a batch dimension if needed
    if len(multicoil_imgs.shape) < 4:
        multicoil_imgs = multicoil_imgs.unsqueeze(0)
        
    b, c, h, w, _ = multicoil_imgs.shape
    
    multicoil_imgs = multicoil_imgs.permute(0,1,4,2,3).reshape(-1, c*2, h, w )
    
    return multicoil_imgs
    

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

#Convert multicoil to singlecoil
def multicoil2single(multicoil_imgs,maps):
    ''' 
    multicoil_imgs: (b, 16, h, w) or (b, 8, h, w, 2)
    '''
    
    #Get the coil images to be complex
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
            #Show coil cobmined estimate (not SENSE)
            S = sp.linop.Multiply((h,w), maps[i])
            combo_img = S.H * tensor_to_complex_np(multicoil_imgs[i].cpu())
            
            combo_imgs.append(to_tensor(combo_img))
        
    combo_imgs = torch.stack(combo_imgs)
        
    return combo_imgs


#Find the sensitivity maps
def get_maps(zf_imgs, num_acs, normalizing_val=None):
    #zf_imgs could be (b, c, h, w), (b, coils, h, w, 2) #, (c, h, w,) or (coils, h, w, 2)
    
    if len(zf_imgs.shape) == 4: #Then (b,c,h,w)
        zf_imgs = chans_to_coils(zf_imgs)
        
    b, num_coils, h, w, _ = zf_imgs.shape
        
    
    #Need to unnormalize zf_imgs before going to kspace
    if normalizing_val is not None:
        zf_imgs = unnorm(zf_imgs, normalizing_val)
    
    #Get the kspace
    masked_kspace = fastmri.fft2c(zf_imgs)
    
    all_maps = []
    
    #Get the mask for each sample in the batch
    for i in range(b):
        maps = mr.app.EspiritCalib(tensor_to_complex_np(masked_kspace[i].cpu()), 
                                    calib_width=num_acs,
                                    show_pbar=False, 
                                    crop=0.70, 
                                    device=sp.Device(0),
                                    kernel_width=6).run().get()
        
        all_maps.append(maps)
        
    all_maps = np.stack(all_maps)
        
    return all_maps

#Unnormalize the values
def unnorm(imgs, normalizing_val):
    #Images should be (b, c, h, w) or (b, coils, h, w, 2)
    if len(imgs.shape) == 4:
        imgs = imgs * normalizing_val.reshape(-1,1,1,1).to(imgs.device)
        
    if len(imgs.shape) == 5:
        imgs = imgs * normalizing_val.reshape(-1,1,1,1,1).to(imgs.device)
        
    return imgs