import torch
import kornia
from typing import Tuple, Any, Callable

from .utils.loader import load_model
from .utils.circle_linear_system import circle_linear_system, const_to_rad


class BoundingCircleDetector():
    device: str
    model: Any
    canny: Callable
    
    def __init__(self, device: str="cuda", name: str="model") -> None:
        self.device = device
        self.model = load_model(device, name)
        self.canny = kornia.filters.Canny()

    def __call__(self, img: torch.FloatTensor, N: int=100, reduction: str="mean") -> Tuple[torch.Tensor, torch.Tensor]:
        """Foward pass of BoundingCircleDetector.

        Args:
            img (torch.FloatTensor): Needs to be normalized in [0, 1].
            N (int): Number of non-zero samples.
            reduction (str): Segmentation reduction along batch dimension.
        Return:
            center (torch.Tensor): Circle's center of shape Bx2.
            radius (torch.Tensor): Circle's radius of shape B.
        """
        if len(img.shape) is not 4:
            raise RuntimeError("BoundingCircleDetector: Expected 4 dimensional input, got {} dimensional input.".format(len(img.shape)))
        if reduction is None:
            seg = self.model(img.to(self.device))
        elif reduction == "mean":
            seg = self.model(img.to(self.device)).mean(dim=0, keepdim=True)
        elif reduction == "max":
            seg, _ = self.model(img.to(self.device)).max(dim=0, keepdim=True)
        else:
            raise ValueError("BoundingCircleDetector: Invalid reduction {} passed.".format(reduction))

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

        center, radius = x[:,:2], const_to_rad(x)

        return center.squeeze(-1), radius.squeeze(-1)
