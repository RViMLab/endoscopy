from typing import Tuple

import torch
from kornia.geometry.transform import get_perspective_transform


def differentiate_duv(duv: torch.Tensor, batch_first: bool = True) -> torch.Tensor:
    r"""Computes the finite difference of duv.

    Args:
        duv (torch.Tensor): Deviation from edges in image coordinates of shape BxTx4x2
        batch_first (bool): If true, expects input of shape BxTx.., else TxBx...

    Return:
        dduv (torch.Tensor): Differentiated duv of shape Bx(T-1)x4x2
    """
    if batch_first:
        dduv = duv.narrow(1, 1, duv.size(1) - 1) - duv.narrow(1, 0, duv.size(1) - 1)
    else:
        dduv = duv.narrow(0, 1, duv.size(0) - 1) - duv.narrow(0, 0, duv.size(0) - 1)
    return dduv


def four_point_homography_to_matrix(
    uv_img: torch.Tensor, duv: torch.Tensor
) -> torch.Tensor:
    r"""Transforms homography from four point representation of shape 4x2 to matrix representation of shape 3x3.
    Args:
        uv_img (torch.Tensor): Image edges in image coordinates of shape ...x4x2
        duv (torch.Tensor): Deviation from edges in image coordinates of shape ...x4x2
    Return:
        h (torch.Tensor): Homography of shape ...x3x3.
    Example:
        h = four_point_homography_to_matrix(uv_img, duv)
    """
    uv_wrp = uv_img + duv
    h = get_perspective_transform(
        uv_img.view((-1,) + uv_img.shape[-2:]).flip(-1),
        uv_wrp.view((-1,) + uv_wrp.shape[-2:]).flip(-1),
    )
    return h.view(uv_img.shape[:-2] + (3, 3))


def frame_pairs(
    video: torch.Tensor, step: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Helper function to return frame pairs at an offset.

    Args:
        video (torch.Tensor): Video clip of shape BxNxCxHxW
        step (int): Number of frames in between image pairs

    Return:
        frames_i (torch.Tensor): Frames starting at time step i with stride step
        frames_ips (torch.Tensor): Frames starting at time step i+step with stride step
    """
    frames_i = video[:, :-step:step]
    frames_ips = video[:, step::step]
    return frames_i, frames_ips


def image_edges(img: torch.Tensor) -> torch.Tensor:
    r"""Returns edges of image (uv) in OpenCV convention.
    Args:
        img (torch.Tensor): Image of shape ...xCxHxW
    Returns:
        uv (torch.Tensor): Image edges of shape ...x4x2
    """
    shape = img.shape[-2:]
    uv = torch.tensor(
        [[0, 0], [0, shape[1]], [shape[0], shape[1]], [shape[0], 0]],
        device=img.device,
        dtype=torch.float32,
    )
    return uv.unsqueeze(0).repeat(img.shape[:-3] + (1, 1))
