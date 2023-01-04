#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 13:48:01 2022

@author: jeff
"""
#%%
import torchvision
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union

from fastmri.data import subsample
from fastmri.data import transforms, mri_data
from fastmri import fftc
from fastmri.evaluate import METRIC_FUNCS, Metrics
import fastmri



#%%

#Show multiple images in a grid from a tensor
def show_tensor_imgs(plot_imgs,nrow = 5, return_grid = False, **kwargs):
    ''' 
    Show tensor images (with values between -1 and 1) in a grid
    
    plot_imgs: (batch_size, num_channels, height, width) [Tensor] tensor of imgs with values between -1 and 1
    nrows: Number of imgs to include in a row
    '''
    
    #Put the images in a grid and show them
    if (plot_imgs[0].dtype == torch.int32) or (plot_imgs[0].dtype == torch.uint8):
        grid = torchvision.utils.make_grid(plot_imgs, nrow = int(nrow), scale_each=False, normalize=False)
        
    else:
        grid = torchvision.utils.make_grid(plot_imgs, nrow = int(nrow), scale_each=True, normalize=True)
        
    if not return_grid:
        f = plt.figure()
        f.set_figheight(15)
        f.set_figwidth(15)
        plt.imshow(grid.permute(1, 2, 0).numpy())
    
        #Use a custom title
        if 'title' in kwargs:
            plt.title(kwargs['title'])

    else:
        return grid



#Create subplots of images
def subplot_imgs(plot_imgs, val_range = [-1.0, 1.0], scale = False, titles = None):

    ''' 
    Show tensor images (with values between -1 and 1) in separate subplots
    
    plot_imgs: (batch_size, num_channels, height, width) [Tensor] tensor of imgs with values between -1 and 1
    titles: list of titles for each subplot
    '''

    batch_size, num_channels, height, width = plot_imgs.size()

    if scale:
        val_range[0] = plot_imgs.min()
        val_range[1] = plot_imgs.max()

    if batch_size == 1:
        plt.figure()
        plt.imshow(plot_imgs[0].permute(1,2,0).numpy(), cmap = plt.get_cmap('gray'), vmin= val_range[0], vmax=val_range[1])

        if titles is not None:
            plt.title(titles)

    else:
        fig, axs = plt.subplots(1, batch_size)

        fig.set_figheight(10)
        fig.set_figwidth(10)

        for i in range(0,batch_size):

            axs[i].imshow(plot_imgs[i].permute(1,2,0).numpy(), cmap = plt.get_cmap('gray'), vmin= val_range[0], vmax=val_range[1])
            
            if titles is not None:
                axs[i].set_title(titles[i])


#Plot the k-space 
def plot_kspace(plot_img, **kwargs):

    plt.figure()
    #Get the log of the magnitude
    magnitude= torch.log(fastmri.complex_abs(plot_img)+ 1e-9)

    #Plot the magnitude
    subplot_imgs(magnitude.unsqueeze(0).unsqueeze(1), scale = True, titles = 'Magnitude')


#Method to normalize put values between 0 and 1
def normalize(x):

    x = (x-x.min())/(x.max()-x.min())

    return x



#Method to get image from k-space matrix
def kspace2img(masked_kspace):
    
    # inverse Fourier transform to get zero filled solution
    image = fastmri.ifft2c(masked_kspace)
    
    image = transforms.complex_center_crop(image, (320,320))

    # absolute value
    image = fastmri.complex_abs(image)


    # normalize input (actually standardizes)
    image, mean, std = transforms.normalize_instance(image, eps=1e-11)
    image = image.clamp(-6, 6)
    
    return image

#Data transform that will give either images or kspace
class GenericDataTransform:
    '''
    A more generic data transform fothe MRI dataset
    Reformated from the fastMRI repository
    '''

    def __init__(self, mask_func: Optional[subsample.MaskFunc]=None, toImg = True, add_transforms = None, use_seed=True):

        self.mask_func = mask_func
        self.toImg = toImg
        self.use_seed = use_seed
        self.add_transforms = add_transforms

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
        
        kspace_torch = transforms.to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            # we only need first element, which is k-space after masking
            masked_kspace, mask = transforms.apply_mask(kspace_torch, self.mask_func, seed=seed)[0:2]
        else:
            masked_kspace = kspace_torch
            mask = None

        if not self.toImg:
            return masked_kspace, mask

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)

        # crop input to correct size
        if target is not None:
            crop_size = (target.shape[-2], target.shape[-1])
        else:
            crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        if image.shape[-2] < crop_size[1]:
            crop_size = (image.shape[-2], image.shape[-2])

        image = transforms.complex_center_crop(image, crop_size)

        # absolute value
        image = fastmri.complex_abs(image)


        # normalize input
        image, mean, std = transforms.normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)
        
        image = image.reshape(1, image.shape[-2], image.shape[-2])
        
        #Additional transformations
        if self.add_transforms is not None:
            image = self.add_transforms(image)
            


        return image, torch.Tensor([0]).unsqueeze(0)
        

#Data transform that will give either images or kspace
class ConditionalDataTransform:
    '''
    A more generic data transform fothe MRI dataset
    Reformated from the fastMRI repository
    '''

    def __init__(self, mask_func: Optional[subsample.MaskFunc]=None, add_transforms = None, use_seed=True):

        self.mask_func = mask_func
        self.use_seed = use_seed
        self.add_transforms = add_transforms

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

        kspace_torch = transforms.to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            # we only need first element, which is k-space after masking
            masked_kspace, mask = transforms.apply_mask(kspace_torch, self.mask_func, seed=seed)[0:2]
        else:
            masked_kspace = kspace_torch
            mask = None


        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)

        # crop input to correct size
        if target is not None:
            crop_size = (target.shape[-2], target.shape[-1])
        else:
            crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        if image.shape[-2] < crop_size[1]:
            crop_size = (image.shape[-2], image.shape[-2])

        image = transforms.complex_center_crop(image, crop_size)

        # absolute value
        image = fastmri.complex_abs(image)


        # normalize input
        image, mean, std = transforms.normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)
        
        image = image.reshape(1, image.shape[-2], image.shape[-2])
        
        #Additional transformations
        if self.add_transforms is not None:
            image = self.add_transforms(image)
            
        
        
        # normalize target
        if target is not None:
            target_torch = transforms.to_tensor(target)
            target_torch = transforms.center_crop(target_torch, crop_size)
            target_torch = transforms.normalize(target_torch, mean, std, eps=1e-11)
            target_torch = target_torch.clamp(-6, 6)
            
            target_torch = target_torch.reshape(1, target_torch.shape[-2], target_torch.shape[-2])
            
            #Additional transformations
            if self.add_transforms is not None:
                target_torch = self.add_transforms(target_torch)
        else:
            target_torch = torch.Tensor([0])


        return target_torch, image

class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

#Get metrics: MSE, NMSE, PSNR, SSIM for a single prediction
def get_metrics(pred, gt):

    #Create the metrics object
    metrics = Metrics(METRIC_FUNCS)
    
    num_pred, _, _, _ = pred.shape

    for i in range(num_pred):
        #Calculate the metrics
        metrics.push(gt[i], pred[i])

    return metrics

#Show absolute error map
def show_error_map(pred, gt, kspace = False, limits = None, title=None):

    err = torch.abs(pred - gt)

    f = plt.figure()
    f.set_figheight(10)
    f.set_figwidth(10)
    
    if kspace:
        err = torch.log(fastmri.complex_abs(pred-gt)+ 1e-9)
        err = err.unsqueeze(0)

    if limits is not None:
        plt.imshow(err.permute(1,2,0).numpy(), cmap = plt.cm.Blues, vmin=limits[0], vmax=limits[1])
    else:
        plt.imshow(err.permute(1,2,0).numpy(), cmap = plt.cm.Blues)
    plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.show()
    
#Show variance map
def show_var_map(var, kspace = False, limits = None, title=None):

    err = torch.abs(var)

    f = plt.figure()
    f.set_figheight(10)
    f.set_figwidth(10)
    
    if limits is not None:
        plt.imshow(err.permute(1,2,0).numpy(), cmap = plt.cm.Blues, vmin=limits[0], vmax=limits[1])
    else:
        plt.imshow(err.permute(1,2,0).numpy(), cmap = plt.cm.Blues)
    plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.show()


def apply_lanczos(img, a = 2, kernel_upsample=4):
    #a determines how many zeros crossings (and lobes) there are
    #Kernel upsample determines the number of points included in the kernel

    #Determine the kernel size and padding
    kernel_size = (a*kernel_upsample*2) + 1
    num_around = (kernel_size-1) / 2
    left_top_padding = int((kernel_size-1) / 2)
    right_bottom_padding = int((kernel_size-1) / 2)
    
    # Need to do a circular convolution, for downsampling by 4, need padding of 1
    img_padded = torch.nn.functional.pad(img.unsqueeze(0), (left_top_padding,right_bottom_padding,left_top_padding,right_bottom_padding),mode='circular')
    
    #Get the kernel values
    kernel_idx = torch.linspace(-num_around / kernel_upsample, num_around / kernel_upsample, kernel_size)
    kernel_1d = torch.sinc(kernel_idx) * torch.sinc(kernel_idx / a)
    
    '''
    plt.figure()
    plt.stem(torch.linspace(-num_around, num_around, kernel_size).numpy(), kernel_1d.numpy()) # to see the full filter
    plt.title('Lanczos Kernel')
    '''
    
    #2D kernel is the multiplication of two kernels for x and y direction
    #Outer product to get the 2D kernel
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    
    filtered_img = torch.nn.functional.conv2d(img_padded, kernel_2d.unsqueeze(0).unsqueeze(0), stride=1).squeeze(0)

    #Need to get values between -1 and 1, since blurring, max and min values need to be found with all ones or minus ones
    filtered_img = (filtered_img - - kernel_2d.sum())/(kernel_2d.sum() - -kernel_2d.sum())
    filtered_img = torchvision.transforms.Normalize(0.5,0.5)(filtered_img)
    
    return filtered_img

#%% Example usage
if __name__ == '__main__':

    #Location of the dataset
    dataset_dir = '../../datasets/fastMRI_brain/data/singlecoil_train'

    #Transform
    data_transform = GenericDataTransform(toImg=True)

    #Load the data 
    dataset = mri_data.SliceDataset(root = dataset_dir, transform = data_transform, challenge='singlecoil')

    #Show the image
    show_tensor_imgs(dataset[0].unsqueeze(0).unsqueeze(1))

    #Transform
    data_transform = GenericDataTransform(toImg=False)

    #Load the data 
    dataset = mri_data.SliceDataset(root = dataset_dir, transform = data_transform, challenge='singlecoil')

    #Show the k-space magnitude
    plot_kspace(dataset[0])

