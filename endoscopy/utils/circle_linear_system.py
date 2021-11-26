import torch
from typing import Tuple


def circle_linear_system(pts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a linear system that represents a circle.

    Args:
        pts (torch.Tensor): BxN points of shape BxNxM, where M >= 2.
    Return:
        A (torch.Tensor): Linear system matrix.
        b (torch.Tensor): Offset to linear equation.
    """
    A = torch.stack(
        [
            2*pts[...,-2],
            2*pts[...,-1],
            torch.ones_like(pts[...,-1])
        ], axis=-1
    )

    b = torch.stack(
        [
            torch.square(pts[...,-2]) + torch.square(pts[...,-1])
        ], axis=-1
    )

    return A, b

def const_to_rad(x: torch.Tensor) -> torch.Tensor:
    """Retrieve radius from solution to linear system.

    Args:
        x (torch.Tensor): Solutions to circle linear system.
    Return:
        radius (torch.Tensor): Circle's radius.
    """
    radius = torch.sqrt(x[:,2] + x[:,0]**2 + x[:,1]**2)
    return radius
