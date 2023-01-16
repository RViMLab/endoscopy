from typing import Any, Tuple

import torch
from kornia.geometry import resize

from .utils.helpers import four_point_homography_to_matrix, image_edges
from .utils.loader import MODEL, load_model


class HomographyPredictor:
    device: str
    model: Any

    def __init__(
        self,
        model: MODEL.HOMOGRAPHY_PREDICTION = MODEL.HOMOGRAPHY_PREDICTION.H_FEATURE_LSTM,
        device: str = "cuda",
    ) -> None:
        self.device = device
        self.model = load_model(model, device)

    def __call__(self) -> None:
        print("hello world")
