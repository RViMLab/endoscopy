import torch
from kornia.geometry.transform import get_perspective_transform


def four_point_homography_to_matrix(uv_img: torch.Tensor, duv: torch.Tensor) -> torch.Tensor:
    r"""Transforms homography from four point representation of shape 4x2 to matrix representation of shape 3x3.
    Args:
        uv_img (torch.Tensor): Image edges in image coordinates
        duv (torch.Tensor): Deviation from edges in image coordinates
    Return:
        h (torch.Tensor): Homography of shape 3x3.
    Example:
        h = four_point_homography_to_matrix(uv_img, duv)
    """
    uv_wrp = uv_img + duv
    h = get_perspective_transform(uv_img.flip(-1), uv_wrp.flip(-1))
    return h


def image_edges(img: torch.Tensor) -> torch.Tensor:
    r"""Returns edges of image (uv) in OpenCV convention.
    Args:
        img (torch.Tensor): Image of shape BxCxHxW
    Returns:
        uv (torch.Tensor): Image edges of shape Bx4x2
    """
    if len(img.shape) != 4:
        raise ValueError("Expected 4 dimensional input, got {} dimensions.".format(len(img.shape)))
    shape = img.shape[-2:]
    uv = torch.tensor(
        [
            [       0,        0],
            [       0, shape[1]],
            [shape[0], shape[1]],
            [shape[0],        0]
        ], device=img.device, dtype=torch.float32
    )
    return uv.unsqueeze(0).repeat(img.shape[0], 1, 1)
