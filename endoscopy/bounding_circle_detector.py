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

    def __call__(self, img: torch.FloatTensor, N: int=100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Foward pass of BoundingCircleDetector.

        Args:
            img (torch.FloatTensor): Needs to be normalized in [0, 1].
            N (int): Number of non-zero samples.
        Return:
            center (torch.Tensor): Circle's center of shape Bx2.
            radius (torch.Tensor): Circle's radius of shape B.
        """
        seg = self.model(img.to(self.device))
        _, edg = self.canny(seg)

        pts = []
        for e in edg:
            if e.numel() < N:
                raise RuntimeError("BoundingCircleDetector: Non suffiecient non-zero elements to sample from, got {}, required {}".format(e.numel(), N))
            nonzero = e.nonzero().float()
            pts.append(
                nonzero[torch.randperm(nonzero.shape[0], device=self.device)[:N]]  # sampling without replacement: https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146/19
            )

        pts = torch.stack(pts)

        A, b = circle_linear_system(pts)
        x = torch.linalg.lstsq(A, b).solution

        center, radius = x[:,:2], const2rad(x)

        return center.squeeze(), radius.squeeze()
