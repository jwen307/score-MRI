import torch
import sigpy as sp
from fastmri.data.transforms import to_tensor, tensor_to_complex_np

def chans_to_coils(multicoil_imgs, training=False):
    # Add a batch dimension if needed
    if len(multicoil_imgs.shape) < 4:
        multicoil_imgs = multicoil_imgs.unsqueeze(0)

    b, c, h, w = multicoil_imgs.shape

    # Split into real and imag
    # if training:
    #    multicoil_imgs = multicoil_imgs.reshape(b, -1, 2, h, w)
    #    multicoil_imgs = multicoil_imgs.permute(0,1,3,4,2)#.contiguous()
    # else:
    multicoil_imgs = torch.stack([multicoil_imgs[:, 0:int(c):2, :, :], multicoil_imgs[:, 1:int(c):2, :, :]], dim=-1)

    return multicoil_imgs


def multicoil2single(multicoil_imgs, maps):
    '''
    multicoil_imgs: (b, 16, h, w) or (b, 8, h, w, 2)
    '''

    # Get the coil images to be complex
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
        # Show coil cobmined estimate (not SENSE)
        S = sp.linop.Multiply((h, w), maps[i])
        combo_img = S.H * tensor_to_complex_np(multicoil_imgs[i].cpu())

        combo_imgs.append(to_tensor(combo_img))

    combo_imgs = torch.stack(combo_imgs)

    return combo_imgs