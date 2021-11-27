import torch
from kornia.geometry.bbox import bbox_generator


def max_rectangle_in_circle(img_shape: torch.Size, center: torch.Tensor, radius: torch.Tensor, safety_margin: int=10, ratio: float=3./4.) -> torch.Tensor:
    r"""Finds the maximum sized rectangle of given aspect ratio within a circle. The circle may be cropped within an image.
    For example see https://drive.google.com/file/d/1GMS1V415pAxdRkf2GoYLWjSFbx33cwew/view?usp=sharing.

    Args:
        img_shape (torch.Size): Shape of image in which circle lies, CxHxW
        center (torch.Tensor): Circle's center of shape Bx2
        radius (torch.Tensor): Circle's radius of shape B
        safety_margin (int): Radius safety margin in pixels
        ratio (float): Height/width ratio of rectanlge
    Return:
        box (torch.Tensor): Box of shape Bx4x2
    """
    if len(center.shape) != 2:
        raise ValueError("Expected center of shape Bx2, got {}".format(center.shape))

    # Construct rectangle of given ratio with edges on circle
    safety_radius = radius - safety_margin
    w = 2*safety_radius/torch.sqrt(torch.tensor([ratio**2 + 1], dtype=safety_radius.dtype, device=safety_radius.device))
    h = torch.sqrt(4*safety_radius**2 - w**2)

    d0 = center
    d1 = torch.tensor(list(img_shape[-2:]), device=center.device).expand_as(center) - center
    d1 -= 1  # shape encodes position + 1

    new_ratio_shape = torch.stack([
        torch.min(2*torch.min(d0[:,0], d1[:,0]) + 1, h),
        torch.min(2*torch.min(d0[:,1], d1[:,1]) + 1, w)
    ], axis=-1)

    new_ratio = (new_ratio_shape[:,0]/new_ratio_shape[:,1]).unsqueeze(-1)

    shape = torch.where(
        ratio/new_ratio < 1,
        torch.stack([ratio*new_ratio_shape[:,1], new_ratio_shape[:,1]], axis=-1),  # update ratio preserving height
        torch.stack([new_ratio_shape[:,0], new_ratio_shape[:,0]/ratio], axis=-1)   # update ratio preserving width
    )

    top_left = torch.stack([
        center[:,0] - (shape[:,0] - 1)/2, center[:,1] - (shape[:,1] - 1)/2
    ], axis=-1)

    return bbox_generator(top_left[:,1], top_left[:,0], shape[:,1], shape[:,0])  # expects x,y coordinates
