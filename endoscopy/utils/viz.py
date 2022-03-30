import numpy as np
import torch
from typing import Union


def yt_alpha_blend(img_y: Union[np.ndarray, torch.Tensor], img_t: Union[np.ndarray, torch.Tensor], alpha: float=.5) -> Union[np.ndarray, torch.Tensor]:
    r"""Blends RGB image into yellow and turquoise.
    Args:
        img_y (np.ndarray or torch.Tensor): Image to be blended in yellow color (np.ndarray: CxHxW, torch.Tensor: ...xHxWxC)
        img_t (np.ndarray or torch.Tensor): Image to be blended in turquoise color (np.ndarray: CxHxW, torch.Tensor: ...xHxWxC)
    Returns:
        blend (np.ndarray or torch.Tensor): Blend of the form alpha*img_y + (1-alpha)*img_t
    """
    if type(img_y) == np.ndarray and type(img_t) == np.ndarray:
        img_y_cpy = img_y.copy() 
        img_t_cpy = img_t.copy()

        img_y_cpy[...,0]  = 0
        img_t_cpy[...,-1] = 0
    elif type(img_y) == torch.Tensor and type(img_t) == torch.Tensor:
        img_y_cpy = img_y.detach().clone()
        img_t_cpy = img_t.detach().clone()

        img_y_cpy[...,0,:,:]  = 0
        img_t_cpy[...,-1,:,:] = 0

    blend = alpha*img_y_cpy + (1-alpha)*img_t_cpy
    return blend
