# -*- coding: utf-8 -*-
import torch
from torch.nn.modules.loss import _Loss, _WeightedLoss
from dival.util.torch_losses import poisson_loss
import fastmri
from .network_utils import chans_to_coils

class CINNNLLLoss(_Loss):
    def __init__(self,  distribution: str, size_average=None, reduce=None, 
                 reduction: str = 'mean') -> None:
        """
        Class for negative log-likelihood loss for cINN models.

        Parameters
        ----------
        distribution : str
            Target distribution for the model:
                'normal': Normal distribution
                'radial': Radial distribution (not yet implemented)
        size_average : TYPE, optional
            DESCRIPTION. 
            The default is None.
        reduce : TYPE, optional
            DESCRIPTION. 
            The default is None.
        reduction : str, optional
            DESCRIPTION. 
            The default is 'mean'.

        Returns
        -------
        None

        """
        super(CINNNLLLoss, self).__init__(size_average, reduce, reduction)
        self.distribution = distribution

    def forward(self, zz, log_jac):
        """
        

        Parameters
        ----------
        zz : TYPE
            DESCRIPTION.
        log_jac : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if self.distribution == 'normal':
            return cinn_normal_nll_loss(zz=zz, log_jac=log_jac,
                                        reduction=self.reduction)
        elif self.distribution == 'radial':
            return cinn_radial_nll_loss(zz=zz, log_jac=log_jac,
                                        reduction=self.reduction)


class CINNBidirectionalLoss(_WeightedLoss):
    def __init__(self, noise, weight=None, size_average=None, reduce=None, 
                 reduction: str = 'mean'):
        """
        

        Parameters
        ----------
        noise : TYPE
            DESCRIPTION.
        weight : TYPE, optional
            DESCRIPTION. The default is None.
        size_average : TYPE, optional
            DESCRIPTION. The default is None.
        reduce : TYPE, optional
            DESCRIPTION. The default is None.
        reduction : str, optional
            DESCRIPTION. The default is 'mean'.

        Returns
        -------
        None.

        """
        super(CINNBidirectionalLoss, self).__init__(weight, size_average, 
                                                    reduce, reduction)
        self.noise = noise

    def forward(self, zz, log_jac, y_pred, y_true):
        """
        

        Parameters
        ----------
        zz : TYPE
            DESCRIPTION.
        log_jac : TYPE
            DESCRIPTION.
        y_pred : TYPE
            DESCRIPTION.
        y_true : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return cinn_bidirectional_loss(zz=zz, log_jac=log_jac, y_pred=y_pred,
                                       y_true=y_true, weight=self.weight,
                                       noise=self.noise, 
                                       reduction=self.reduction)


def cinn_normal_nll_loss(zz, log_jac, reduction='mean'):
    """
    Negative log-likelihood loss for a cINN model with a normal distribution
    as target.

    Parameters
    ----------
    zz : torch tensor
        Vector from the normal distribution.
    log_jac : torch tensor
        Log det of the Jacobian.

    Returns
    -------
    torch tensor
        NLL score for zz.

    """

    ndim_total = zz.shape[-1]
    c =  ndim_total / 2. * torch.log(torch.tensor(2.0*3.14159))
    #print(torch.mean(zz**2) / 2)
    #if torch.is_tensor(log_jac):
    ret = c / ndim_total + torch.mean(zz**2) / 2 - torch.mean(log_jac) / ndim_total
    #else: 
    #    ret = c / ndim_total + torch.mean(zz**2) / 2 # log_jac = 0 or constant

    ret = ret / torch.log(torch.tensor(2.)) + 8.
    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret


def cinn_radial_nll_loss(zz, log_jac, reduction='mean'):
    """
    Negative log-likelihood loss for a cINN model with a radial distribution
    as target.

    Parameters
    ----------
    zz : torch tensor
        Vector from the radial distribution.
    log_jac : torch tensor
        Log det of the Jacobian.

    Returns
    -------
    torch tensor
        NLL score for zz.

    """
    ndim_total = zz.shape[-1]
    ret = (ndim_total - 1) * torch.mean(torch.log(torch.norm(zz, dim=1))) / ndim_total + \
          torch.mean(zz**2) / 2 - torch.mean(log_jac) / ndim_total
    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret    

def cinn_bidirectional_loss(zz, log_jac, y_pred, y_true, weight, distribution,
                            noise, reduction='mean', **kwargs):
    """
    Compute a bidirectional loss for the cINN model. It consists of the NLL
    loss and a data consistency term.

    Parameters
    ----------
    zz : torch tensor
        Vector from the normal distribution.
    log_jac : torch tensor
        Log of the Jacobian..
    y_pred : torch tensor
        Predicted measurement.
    y_true : torch tensor
        True (noisy) measurement.
    weight : torch tensor
        Weighting of the NLL.
    noise : str
        Noise model on the measurements: "poisson" or "gaussian".

    Returns
    -------
    torch tensor
        Weighted loss of the data consistency and NLL.

    """
    
    if zz is not None:
        if distribution == 'normal':
            nll = cinn_normal_nll_loss(zz=zz, log_jac=log_jac, reduction=None)
        if distribution == 'radial':
            nll = cinn_radial_nll_loss(zz=zz, log_jac=log_jac, reduction=None)
    else:
        nll = 0.
        
    if y_pred is not None:
        if noise == 'poisson':
            y_loss = poisson_loss(y_pred=y_pred, y_true=y_true, **kwargs)
        elif noise == 'gaussian':
            y_loss = torch.nn.functional.mse_loss(input=y_pred, target=y_true, 
                                                  **kwargs)
        else:
            raise ValueError("Noise distribution must be', "
                         "'poisson' or 'gaussian', not '{}'".format(noise))
    else:
        y_loss = 0.
        
    ret = y_loss + weight*nll    
        
    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret


class FourierMSELoss(torch.nn.Module):
    
    def __init__(self) -> None:
        super(FourierMSELoss, self).__init__()
        
    def forward(self, x: torch.Tensor, target:torch.Tensor, mask:torch.Tensor, std:torch.Tensor) -> torch.Tensor:
        
        #Calculate the different images (Note: only need std since mean gets cancelled)
        diff_img = std.reshape(-1,1,1,1)*(target - x)
        
        #Add zero channel if real image
        if target.shape[1] < 2:
            diff_img = torch.cat([diff_img, torch.zeros_like(diff_img)],dim=1)
        
        #Find the Fourier transforms of the difference image
        f_diff = fastmri.fft2c(diff_img.permute(0,2,3,1))
        
        #Invert the mask to only account for unknown Fourier components
        mask = mask*-1 + 1
        
        #Apply the mask (starts as (batch_size, 1, num_rows))
        mask2D = mask.permute(0,2,1).repeat(1,mask.size()[1],1) #Make the mask 2D for removing columns
        f_diff_masked = mask2D.unsqueeze(-1).repeat(1,1,1,2) * f_diff
        
        #Calculate the squared error
        loss = torch.mean(torch.square(torch.norm(f_diff_masked, p ='fro')))

        return loss
    
class DCLossMulticoil(torch.nn.Module):
    
    def __init__(self) -> None:
        super(DCLossMulticoil, self).__init__()
        
    def forward(self, x: torch.Tensor, target:torch.Tensor, mask:torch.Tensor, std:torch.Tensor) -> torch.Tensor:
        
        #Calculate the different images (Note: only need std since mean gets cancelled)
        diff_img = std.reshape(-1,1,1,1)*(target - x)
        
        #Convert so there's two channels and 8 coils
        diff_img = chans_to_coils(diff_img, training=True)
        
        #Add zero channel if real image
        #if target.shape[1] < 2:
        #    diff_img = torch.cat([diff_img, torch.zeros_like(diff_img)],dim=1)
        
        #Find the Fourier transforms of the difference image
        f_diff = fastmri.fft2c(diff_img)
        
        #Invert the mask to only account for unknown Fourier components
        mask = mask*-1 + 1
        
        #Apply the mask (starts as (batch_size, 1, num_rows))
        mask2D = mask[0].permute(0,2,1).repeat(1,mask.shape[2],1) #Make the mask 2D for removing columns
        f_diff_masked = mask2D.unsqueeze(-1).unsqueeze(0).repeat(1,1,1,1,2) * f_diff
        
        #Calculate the squared error
        loss = torch.mean(torch.square(torch.norm(f_diff_masked, p ='fro')))

        return loss
    
class KSpaceLoss(torch.nn.Module):
    
    def __init__(self) -> None:
        super(KSpaceLoss, self).__init__()
        
    def forward(self, x: torch.Tensor, target:torch.Tensor, mask:torch.Tensor, std:torch.Tensor) -> torch.Tensor:
        
        #Calculate the different images (Note: only need std since mean gets cancelled)
        diff_img = std.reshape(-1,1,1,1)*(target - x)
        
        #Convert so there's two channels and 8 coils
        diff_img = chans_to_coils(diff_img, training=True)
        
        #Add zero channel if real image
        #if target.shape[1] < 2:
        #    diff_img = torch.cat([diff_img, torch.zeros_like(diff_img)],dim=1)
        
        #Find the Fourier transforms of the difference image
        f_diff = fastmri.fft2c(diff_img)
        
        #Calculate the squared error
        loss = torch.mean(torch.square(torch.norm(f_diff, p ='fro')))

        return loss