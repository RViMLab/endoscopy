import torch
import kornia
from typing import Tuple, Any, Callable

from .utils.loader import load_model
from .utils.circle_linear_system import circle_linear_system, const2rad


class BoundingCircleDetector():
    device: str
    model: Any
    canny: Callable
    
    def __init__(self, device: str="cuda", name: str="model") -> None:
        self.device = device
        self.model = load_model(device, name)
        self.canny = kornia.filters.Canny()

    def __call__(self, img: torch.FloatTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Foward pass of BoundingCircleDetector.

        Args:
            img (torch.FloatTensor): Needs to be normalized in [0, 1].
        Return:
            center (torch.Tensor): Circle's center.
            radius (torch.Tensor): Circle's radius.
        """
        seg = self.model(img.to(self.device))
        _, edg = self.canny(seg)

        pts = edg.nonzero().float()

        A, b = circle_linear_system(pts)
        x = torch.linalg.lstsq(A, b).solution

        center, radius = x[:2], const2rad(x)

        return center, radius


