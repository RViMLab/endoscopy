import cv2
import numpy as np
from typing import Tuple


def boundaryRectangle(img: np.ndarray, th: int=10) -> Tuple[np.ndarray, tuple]:
    r"""Finds the rectangle that circumferences an endoscopic image.

    Args:
        img (np.ndarray): Grayscale image of shape HxW
        th (int): Whiten threshold, each pixel where value > th is whitened

    Return:
        rectangle (Tuple[np.ndarray, tuple]): Top left corner and shape of found rectangle
    """
    img = np.where(img < th, 0, 255).astype(np.uint8)
    
    col_mean = img.mean(axis=0)
    row_mean = img.mean(axis=1)

    nonzero_col_mean = np.nonzero(col_mean)
    nonzero_row_mean = np.nonzero(row_mean)

    if not len(nonzero_col_mean[0]) or not len(nonzero_row_mean[0]):
        return np.array([]), tuple((0,))

    top    = np.min(nonzero_row_mean)
    bottom = np.max(nonzero_row_mean)
    left   = np.min(nonzero_col_mean)
    right  = np.max(nonzero_col_mean)

    top_left = np.array([top, left])
    shape = (bottom - top + 1, right - left + 1)

    return top_left, shape


def boundaryCircle(img: np.ndarray, th: int=10) -> Tuple[np.ndarray, float]:
    r"""Find the circle that circumferences an endoscopic image. Works only with full view of the endoscopic image.

    Args:
        img (np.ndarray): Grayscale image of shape HxW
        th (int): Whiten threshold, each pixel where value > th is whitened

    Return:
        circle (Tuple[np.ndarray, float]): Center and radius of found circle
    """
    img = np.where(img < th, 0, 255).astype(np.uint8)

    col_mean = img.mean(axis=0)
    row_mean = img.mean(axis=1)

    col_com = np.sum(np.multiply(np.arange(col_mean.shape[0]), col_mean), axis=0)/col_mean.sum()
    row_com = np.sum(np.multiply(np.arange(row_mean.shape[0]), row_mean), axis=0)/row_mean.sum()

    nonzero_col_mean = np.nonzero(col_mean)
    nonzero_row_mean = np.nonzero(row_mean)

    if not len(nonzero_col_mean[0]) or not len(nonzero_row_mean[0]):
        return np.array([]), None

    col_radius = (np.max(nonzero_col_mean) - np.min(nonzero_col_mean))/2.
    row_radius = (np.max(nonzero_row_mean) - np.min(nonzero_row_mean))/2.

    radius = max(col_radius, row_radius)

    return np.array([row_com, col_com]), radius


if __name__ == '__main__':
    import os

    prefix = os.getcwd()
    in_file = 'data/eye.tif'

    img = cv2.imread(os.path.join(prefix, in_file))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    top_left, shape = boundaryRectangle(img_gray, th=30)   
    center, radius = boundaryCircle(img_gray, th=30)

    top_left, shape, center, radius = top_left.astype(np.int), tuple(map(int, shape)), center.astype(np.int), int(radius)

    cv2.rectangle(img, (top_left[1], top_left[0]), (top_left[1] + shape[1], top_left[0] + shape[0]), (255, 255, 0), 1)
    cv2.circle(img, (center[1], center[0]), radius, (0,255,255), 1)
    cv2.circle(img, (center[1], center[0]), 2, (255,0,255), 4)

    cv2.imshow('img', img)
    cv2.waitKey()
