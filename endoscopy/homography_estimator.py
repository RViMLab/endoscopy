from typing import Any, Tuple

import torch
from kornia.geometry import resize

from .utils.helpers import four_point_homography_to_matrix, image_edges
from .utils.loader import MODEL, load_model


class HomographyEstimator:
    device: str
    model: Any

    def __init__(
        self,
        model: MODEL.HOMOGRAPHY_ESTIMATION = MODEL.HOMOGRAPHY_ESTIMATION.H_48_RESNET_34,
        device: str = "cuda",
    ) -> None:
        self.device = device
        self.model = load_model(model, device)

    def __call__(
        self, img: torch.FloatTensor, wrp: torch.FloatTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Foward pass of BoundingCircleDetector.

        Args:
            img (torch.FloatTensor): Needs to be normalized in [0, 1]. Will be resized to 240x320.
            wrp (torch.FloatTensor): Needs to be normalized in [0, 1]. Will be resized to 240x320.
        Return:
            h (torch.Tensor): Homography of shape Bx3x3
            duv (torch.Tensor): Four point homography of shape Bx4x2
        """
        if img.dim() != 4 or wrp.dim() != 4:
            raise ValueError(
                f"BoundingCircleDetector: Expected 4 dimensional input, got {img.dim()} dimensional input."
            )

        img, wrp = resize(img, [240, 320]), resize(wrp, [240, 320])

        with torch.no_grad():
            duv = self.model(img.to(self.device), wrp.to(self.device))
        uv_img = image_edges(img).to(self.device)
        h = four_point_homography_to_matrix(uv_img, duv)

        return h.to(img.device), duv.to(img.device)
