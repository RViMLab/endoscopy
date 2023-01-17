from typing import Any, Tuple

import torch
from kornia.geometry import resize

from .utils.helpers import (
    differentiate_duv,
    four_point_homography_to_matrix,
    frame_pairs,
    image_edges,
)
from .utils.loader import MODEL, load_model


class HomographyPredictor:
    device: str
    estimator: Any
    predictor: Any

    def __init__(
        self,
        estimator: MODEL.HOMOGRAPHY_ESTIMATION = MODEL.HOMOGRAPHY_ESTIMATION.H_48_RESNET_34,
        predictor: MODEL.HOMOGRAPHY_PREDICTION = MODEL.HOMOGRAPHY_PREDICTION.H_FEATURE_LSTM,
        device: str = "cuda",
    ) -> None:
        self.device = device
        self.estimator = load_model(estimator, device)
        self.predictor = load_model(predictor, device)

    def __call__(
        self, imgs: torch.FloatTensor, increment: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Predicts future homographies given the past imgs.

        Args:
            imgs (torch.FloatTensor): Image sequence of shape BxTxCxHxW. Will be resized to 240x320.
            increment (int): Increment between frame pairs.

        Return:
            hs_ip1 (torch.Tensor): Homography of shape Bx3x3
            duvs_ip1 (torch.Tensor): Four point homography of shape Bx4x2
        """
        if imgs.dim() != 5:
            raise ValueError(
                f"HomographyPredictor: Expected 5 dimensional input, got {imgs.dim()} dimensional input."
            )
        if imgs.shape[1] < 3:
            raise ValueError(
                f"At least 3 images required to perform homography prediction, got {imgs.shape[1]} images."
            )
        if increment < 1:
            raise ValueError(f"Expected increment greated equal to 1, got {increment}.")

        imgs = resize(imgs, [240, 320])
        B, T, C, H, W = imgs.shape
        imgs, imgs_ip1 = frame_pairs(imgs, increment)

        with torch.no_grad():
            imgs, imgs_ip1 = imgs.view(-1, C, H, W), imgs_ip1.view(
                -1, C, H, W
            )  # Bx(T-1)xCxHxW -> B*(T-1)xCxHxW
            duvs = self.estimator(imgs.to(self.device), imgs_ip1.to(self.device))
            duvs = duvs.view(B, T - 1, 4, 2)  # B*(T-1)x4x2 -> Bx(T-1)x4x2
            dduvs = differentiate_duv(duvs, batch_first=True)  # Bx(T-2)x4x2
            imgs, imgs_ip1 = imgs.view(B, T - 1, C, H, W), imgs_ip1.view(
                B, T - 1, C, H, W
            )  # B*(T-1)xCxHxW -> Bx(T-1)xCxHxW
            duvs_ip1 = self.predictor(
                imgs_ip1[:, 1:].to(self.device), duvs[:, 1:], dduvs
            ).to(imgs.device)
        h_ip1 = four_point_homography_to_matrix(
            image_edges(imgs[:, 1:]).to(imgs.device), duvs_ip1
        )

        return h_ip1, duvs_ip1
