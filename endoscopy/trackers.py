import numpy as np
from typing import List, Tuple

from .boundary_detection import boundaryCircle, boundaryRectangle
from .processing import illuminationLevel


class CoMBoundaryTracker(object):
    r"""Center of mass boundary tracker with memory of previous values and
    illumination level check.
    """
    def __init__(self):
        self._center, self._radius = np.array([]), None
        self._top_left, self._shape = np.array([]), tuple()

    def updateBoundaryCircle(self, img: np.ndarray, th1: int=10, th2: float=0.98):
        r"""Update the boundary circle.

        Args:
            img (np.ndarray): Grayscale image or segmentation mask of shape HxW
            th1 (int): Gradient threshold, only look for gradient where mean < th
            th2 (float): Illumination level threshold in [0, 1]. If illumination >= th2, update boundary rectanlge          

        Return:
            circle (Tuple[np.ndarray, float]): Center and radius of found circle, or previous values
        """
        center, radius = boundaryCircle(img, th1)
        if center.shape[0] == 0 or radius is None:
            return self._center, self._radius
            
        illumination = illuminationLevel(img, center, radius)

        if illumination >= th2:
            self._center, self._radius = center, radius

        if self._center.shape[0] == 0 or self._radius is None:  # not initialized
            return self._center, self._radius

        return self._center.astype(np.int), int(self._radius)

    def updateBoundaryRectangle(self, img: np.ndarray, th1: int=10, th2: float=0.98) -> Tuple[np.ndarray, tuple]:
        r"""Update the boundary rectanlge.

        Args:
            img (np.ndarray): Grayscale image or segmentation mask of shape HxW
            th1 (int): Gradient threshold, only look for gradient where mean < th
            th2 (float): Illumination level threshold in [0, 1]. If illumination >= th2, update boundary rectanlge
            
        Return:
            rectanlge (Tuple[np.ndarray, tuple]): Top left corner and shape of found rectangle, or previous values
        """
        top_left, shape = boundaryRectangle(img, th1)
        center, radius = np.array([top_left[0] + shape[0]/2, top_left[1] + shape[1]/2], dtype=np.int), int(max(shape)/2)
        illumination = illuminationLevel(img, center, radius)

        if illumination >= th2:
            self._top_left, self._shape = top_left, shape

        return self._top_left, self._shape

    @property
    def circle(self) -> Tuple[np.ndarray, float]:
        return self._center, self._radius

    @circle.setter
    def circle(self, center: np.ndarray, radius: float) -> None:
        self._center, self._radius = center, radius

    @property
    def rectanlge(self) -> Tuple[np.ndarray, tuple]:
        return self._top_left, self._shape

    @rectanlge.setter
    def rectangle(self, top_left: np.ndarray, shape: tuple) -> None:
        self._top_left, self._shape = top_left, shape
