import cv2
import numpy as np
from typing import List

from .processing import binaryAvg, binaryVar


class ImageBuffer():
    def __init__(self, buffer_size: int=1) -> None:
        self._buffer_size = buffer_size        
        if buffer_size <= 0:
            raise ValueError("Buffer buffer_size must be greater than zero.")
        self._buffer = []

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    @buffer_size.setter
    def buffer_size(self, buffer_size: np.uint8) -> None:
        if buffer_size <= 0:
            raise ValueError("Buffer buffer_size must be greater than zero.")
        else:
            del self._buffer
            self._buffer_size = buffer_size

    @property
    def buffer(self) -> List[np.ndarray]:
        return self._buffer

    @buffer.deleter
    def buffer(self) -> None:
        del self._buffer

    def binaryAvg(self, th: float) -> np.ndarray:
        r"""Averages buffer and return binary image with cut-off threshold th.

        Args:
            th (float): After averaging the buffer, everything below th is set to 0, else 255

        Return:
            avg (np.ndarray): Binary averaged buffer
        """
        return binaryAvg(self._buffer, th)

    def binaryVar(self, th: float) -> np.ndarray:
        r"""Averages buffer and return binary image with cut-off threshold th.

        Args:
            th (float): After computing the buffer's variance, everything below th is set to 255, else 0

        Return:
            var (np.ndarray): Binary variance
        """
        return binaryVar(self._buffer, th)
    
    def appendBuffer(self, img: np.ndarray, conversion: str='BGR2GRAY') -> None:
        r"""Takes image and appends buffer with grayscale of that image.

        Args:
            img (np.ndarray): Image of shape HxWxC
            conversion (str): OpenCV style conversions
        """
        if conversion == 'BGR2GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif conversion == 'RGB2GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError('Requested conversion "{}" not implemented.'.format(conversion))
        self._buffer.append(img)

        if len(self._buffer) > self._buffer_size:
            self._buffer.pop(0)


if __name__ == '__main__':
    import os
    import cv2

    prefix = os.getcwd()
    file = 'data/endo.mp4'

    vr = cv2.VideoCapture(os.path.join(prefix, file))

    ib = ImageBuffer(buffer_size=25)
 
    while vr.isOpened():

        _, img = vr.read()
        if img is None:
            break

        ib.appendBuffer(img)
        # img = ib.binaryAvg(5)
        img = ib.binaryVar(0.5)

        cv2.imshow('img', img)
        cv2.waitKey()
