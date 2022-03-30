import torch
from typing import Tuple, Any

from .utils.loader import MODEL, load_model
from .utils.helpers import four_point_homography_to_matrix, image_edges


class HomographyEstimator():
    device: str
    model: Any
    
    def __init__(self, device: str="cuda", model: MODEL.HOMOGRAPHY_ESTIMATION=MODEL.HOMOGRAPHY_ESTIMATION.RESNET_34) -> None:
        self.device = device
        self.model = load_model(device, model)

    def __call__(self, img: torch.FloatTensor, wrp: torch.FloatTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Foward pass of BoundingCircleDetector.

        Args:
            img (torch.FloatTensor): Needs to be normalized in [0, 1].
            wrp (torch.FloatTensor): Needs to be normalized in [0, 1].
        Return:
            h (torch.Tensor): Homography of shape Bx3x3
            duv (torch.Tensor): Four point homography of shape Bx4x2
        """
        if img.dim() != 4 or wrp.dim() != 4:
            raise RuntimeError("BoundingCircleDetector: Expected 4 dimensional input, got {} dimensional input.".format(img.dim()))

        duv = self.model(img, wrp)
        uv_img = image_edges(img)
        h = four_point_homography_to_matrix(uv_img, duv)

        return h, duv
