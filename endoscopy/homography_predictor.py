from typing import Any, Tuple

import torch
from kornia.geometry import resize

from .utils.helpers import (
    four_point_homography_to_matrix,
    image_edges,
)
from .utils.loader import MODEL, load_model


class HomographyPredictor:
    device: str
    predictor: Any
    hidden_size: int

    def __init__(
        self,
        predictor: MODEL.HOMOGRAPHY_PREDICTION = MODEL.HOMOGRAPHY_PREDICTION.RESNET_34_IN_27,
        device: str = "cuda",
    ) -> None:
        self.device = device
        self.predictor = load_model(predictor, device)
        self.in_channels = None
        if (
            predictor == MODEL.HOMOGRAPHY_PREDICTION.RESNET_34_IN_27
            or predictor == MODEL.HOMOGRAPHY_PREDICTION.RESNET_50_IN_27
        ):
            self.in_channels = 27

    def __call__(
        self,
        imgs: torch.FloatTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Predicts future homographies given the past imgs.

        Args:
            imgs (torch.FloatTensor): Image sequence of shape Bx{self.in_channels / C}xCxHxW. Will be resized to 240x320.

        Return:
            h (torch.Tensor): Homography of shape Bx3x3
            duv (torch.Tensor): Four point homography of shape Bx4x2
        """
        if imgs.dim() != 5:
            raise ValueError(
                f"HomographyPredictor: Expected 5 dimensional input, got {imgs.dim()} dimensional input."
            )
        if imgs.shape[2] != 3:
            raise ValueError(
                f"HomographyPredictor: Expected image with 3 channels, got {imgs.shape[2]}."
            )
        if imgs.shape[1] != int(self.in_channels / 3):
            raise ValueError(
                f"HomographyPredictor: Expected {int(self.in_channels / 3)} images, got {imgs.shape[1]} images."
            )
        tmp_device = imgs.device
        imgs = resize(imgs, [240, 320]).to(self.device)
        B, T, C, H, W = imgs.shape
        imgs = imgs.reshape(B, T * C, H, W)

        with torch.no_grad():
            duv = self.predictor(imgs)
            duv = duv.view(B, 4, 2)
            imgs = imgs.view(B, T, C, H, W)
            h = four_point_homography_to_matrix(image_edges(imgs)[:, 0], duv)

        return (
            h.to(tmp_device),
            duv.to(tmp_device),
        )
