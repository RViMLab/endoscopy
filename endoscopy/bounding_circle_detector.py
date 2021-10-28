import torch
from .utils.loader import load_model
# from .utils.post_processing import 

import torch
from typing import Any


class BoundingCircleDetector():
    device: str
    model: Any
    
    def __init__(self, device: str="cuda", name: str="model") -> None:
        self.device = device
        self.model = load_model(device, name)
        # print(self.model.type)

    def __call__(self, img: torch.FloatTensor) -> torch.FloatTensor:
        """Foward pass of BoundingCircleDetector.

        Args:
            img (torch.FloatTensor): Needs to be normalized in [0, 1].
        Return:
            seg (torch.FloatTensor): Segmentation mask.
        """
        seg = self.model(img.to(self.device))
        return seg


