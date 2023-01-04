import matplotlib.pyplot as plt
import numpy as np
import torch
from network_utils import chans_to_coils
import sigpy as sp
import importlib
from typing import Optional
import torchvision
import fastmri
import sigpy as sp
import sigpy.mri as mr
from fastmri.data.transforms import to_tensor, tensor_to_complex_np
import sigpy.mri as mr
import matplotlib as mpl


def show_multicoil_combo(multicoil_imgs, maps, val_range=None, return_grid=False, **kwargs):
    # Get the coil images from the channels
    if multicoil_imgs.shape[-1] != 2:
        multicoil_imgs = chans_to_coils(multicoil_imgs)

    b, num_coils, h, w, _ = multicoil_imgs.shape

    if torch.is_tensor(maps):
        if len(maps.shape) < 4:
            maps = maps.unsqueeze(0)
        maps = maps.cpu().numpy()
    else:
        if len(maps.shape) < 4:
            maps = np.expand_dims(maps, axis=0)

    combo_imgs = []
    for i in range(b):
        #with sp.Device(0):
            # Show SENSE estimate
            S = sp.linop.Multiply((h, w), maps[i])
            combo_img = S.H * tensor_to_complex_np(multicoil_imgs[i].cpu())

            combo_imgs.append(to_tensor(combo_img))

    combo_imgs = torch.stack(combo_imgs)

    if val_range is None:

        if return_grid:
            grid = show_img(combo_imgs.cpu(), return_grid=True, **kwargs)

        else:
            show_img(combo_imgs.cpu(), return_grid=False, **kwargs)

    else:
        if return_grid:
            grid = show_img(combo_imgs.cpu(), val_range=val_range, return_grid=True, **kwargs)

        else:
            show_img(combo_imgs.cpu(), val_range=val_range, return_grid=False, **kwargs)

    if return_grid:
        return grid


def show_img(plot_imgs, nrow=5, return_grid=False, mean=0, std=1, colormap=None, scale_each=True, **kwargs):
    '''
    Show any type of image (real or complex)
    '''

    # Check if it is a np array
    if isinstance(plot_imgs, np.ndarray):
        plot_imgs = torch.from_numpy(plot_imgs)

    # Put the images on the cpu
    plot_imgs = plot_imgs.detach().cpu()

    # Make the image 4 dims
    if len(plot_imgs.shape) == 3:
        plot_imgs = plot_imgs.unsqueeze(0)

    # Put the channel dimensions in the second dimension if needed
    if plot_imgs.shape[-1] < 5:
        plot_imgs = plot_imgs.permute(0, 3, 1, 2)

    # Get the magnitude image if complex
    if plot_imgs.shape[1] == 2:
        plot_imgs = plot_imgs.permute(0, 2, 3, 1)
        # plot_imgs = fastmri.complex_abs(plot_imgs*(std + 1e-11) + mean).unsqueeze(1)
        plot_imgs = fastmri.complex_abs(plot_imgs).unsqueeze(1)

    # Put the images in a grid and show them
    if (plot_imgs[0].dtype == torch.int32) or (plot_imgs[0].dtype == torch.uint8):
        grid = torchvision.utils.make_grid(plot_imgs, nrow=int(nrow), scale_each=False, normalize=False)

    elif 'val_range' in kwargs:
        grid = torchvision.utils.make_grid(plot_imgs, nrow=int(nrow), normalize=True, value_range=kwargs['val_range'])

    elif colormap is not None:
        grid = torchvision.utils.make_grid(plot_imgs, nrow=int(nrow), scale_each=False, normalize=False)

    else:
        grid = torchvision.utils.make_grid(plot_imgs, nrow=int(nrow), scale_each=scale_each, normalize=True)

    if 'save_dir' in kwargs:

        if 'rect' in kwargs:
            f = plt.figure()
            f.set_figheight(15)
            f.set_figwidth(15)

            if 'rect' in kwargs:
                for i, rect in enumerate(kwargs['rect']):
                    plt.gca().add_patch(
                        Rectangle((rect[0], rect[1]), rect[2], rect[3], linewidth=5, edgecolor=kwargs['rect_colors'][i],
                                  facecolor='none'))
                    # grid = draw_bounding_boxes(grid, kwargs['rect'], colors=kwargs['rect_colors'])

            plt.axis("off")

            if colormap is not None:
                plt.imshow(grid[0].unsqueeze(0).permute(1, 2, 0).numpy(), cmap='RdGy', vmin=grid.min(), vmax=grid.max())
            else:
                plt.imshow(grid.permute(1, 2, 0).numpy())

            # Use a custom title
            if 'title' in kwargs:
                plt.title(kwargs['title'])

            # Save the image
            plt.savefig(kwargs['save_dir'], bbox_inches='tight', pad_inches=0)

        else:
            # Save the image
            torchvision.utils.save_image(grid, kwargs['save_dir'])

    else:
        if not return_grid:
            f = plt.figure()
            f.set_figheight(15)
            f.set_figwidth(15)

            if 'rect' in kwargs:
                for i, rect in enumerate(kwargs['rect']):
                    plt.gca().add_patch(
                        Rectangle((rect[0], rect[1]), rect[2], rect[3], linewidth=2, edgecolor=kwargs['rect_colors'][i],
                                  facecolor='none'))
                    # grid = draw_bounding_boxes(grid, kwargs['rect'], colors=kwargs['rect_colors'])

            plt.axis("off")

            if colormap is not None:
                plt.imshow(grid[0].unsqueeze(0).permute(1, 2, 0).numpy(), cmap='RdGy', vmin=grid.min(), vmax=grid.max())
            else:
                plt.imshow(grid.permute(1, 2, 0).numpy())

            # Use a custom title
            if 'title' in kwargs:
                plt.title(kwargs['title'])

        else:
            return grid
        
        
        
def show_std_map(plot_imgs, kspace = False, limits = None, title=None, **kwargs ):
    
    #Check if it is a np array
    if isinstance(plot_imgs, np.ndarray):
        plot_imgs = torch.from_numpy(plot_imgs)
        
    #Put the images on the cpu
    plot_imgs = plot_imgs.detach().cpu()
    
    #Make the image 4 dims
    if len(plot_imgs.shape) == 3:
        plot_imgs = plot_imgs.unsqueeze(0)
        
    #Put the channel dimensions in the second dimension if needed
    if plot_imgs.shape[-1] < 5:
        plot_imgs = plot_imgs.permute(0,3,1,2)
        
    #Get the magnitude image if complex
    if plot_imgs.shape[1] == 2:
        plot_imgs = plot_imgs.permute(0,2,3,1)
        #plot_imgs = fastmri.complex_abs(plot_imgs*(std + 1e-11) + mean).unsqueeze(1)
        plot_imgs = fastmri.complex_abs(plot_imgs).unsqueeze(1)
        
    err = plot_imgs.std(dim=0)
    #err = torch.abs(var)

    f = plt.figure()
    f.set_figheight(10)
    f.set_figwidth(10)
    
    if 'rect' in kwargs:
        for i, rect in enumerate(kwargs['rect']):
            plt.gca().add_patch(Rectangle((rect[0],rect[1]),rect[2],rect[3],linewidth=5,edgecolor=kwargs['rect_colors'][i],facecolor='none'))        
        #grid = draw_bounding_boxes(grid, kwargs['rect'], colors=kwargs['rect_colors'])
    
    if limits is not None:
        im = plt.imshow(err.permute(1,2,0).numpy(), cmap = mpl.colormaps['viridis'], vmin=limits[0], vmax=limits[1])
    else:
        im = plt.imshow(err.permute(1,2,0).numpy(), cmap = mpl.colormaps['viridis'])
        
    if not 'no_colorbar' in kwargs: 
        plt.colorbar(fraction=0.045, pad=0.01)
        
    plt.margins(x=0.01,y=0.01)
    
    
    if title is not None:
        plt.title(title)
    plt.axis('off')
    
    if not 'no_colorbar' in kwargs: 
        im.figure.axes[1].tick_params(axis="y", labelsize=21)
    
    if 'save_dir' in kwargs:
        #Save the image
        plt.savefig(kwargs['save_dir'], bbox_inches='tight',pad_inches=0)
        
    else:
        plt.show()