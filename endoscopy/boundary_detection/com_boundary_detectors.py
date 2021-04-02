import cv2
import numpy as np
from typing import Tuple


def boundaryRectangle(img: np.ndarray, th: int=10) -> Tuple[np.ndarray, tuple]:
    r"""Finds the rectangle that circumferences an endoscopic image.

    Args:
        img (np.ndarray): Grayscale image of shape HxW
        th (int): Gradient threshold, only look for gradient where mean < th

    Return:
        rectangle (Tuple[np.ndarray, tuple]): Top left corner and shape of found rectangle
    """
    col_mean = img.mean(axis=0)
    row_mean = img.mean(axis=1)

    col_grad = np.gradient(col_mean)
    row_grad = np.gradient(row_mean)

    col_grad = np.where(col_mean < th, col_grad, 0)  # search gradient where mean < th
    row_grad = np.where(row_mean < th, row_grad, 0)

    top    = np.argmax(row_grad)
    bottom = np.argmin(row_grad)
    left   = np.argmax(col_grad)
    right  = np.argmin(col_grad)

    top_left = np.array([top, left])
    shape = (bottom - top + 1, right - left + 1)

    return top_left, shape


def boundaryCircle(img: np.ndarray, th: int=10) -> Tuple[np.ndarray, float]:
    r"""Find the circle that circumferences an endoscopic image. Works only with full view of the endoscopic image.

    Args:
        img (np.ndarray): Grayscale image of shape HxW
        th (int): Gradient threshold, only look for gradient where mean < th

    Return:
        circle (Tuple[np.ndarray, float]): Center and radius of found circle
    """
    col_mean = img.mean(axis=0)
    row_mean = img.mean(axis=1)

    col_grad = np.gradient(col_mean)
    row_grad = np.gradient(row_mean)

    col_grad = np.where(col_mean < th, col_grad, 0)  # search gradient where mean < th
    row_grad = np.where(row_mean < th, row_grad, 0)

    # min/max
    col_min, col_max = np.argmin(col_grad), np.argmax(col_grad)
    row_min, row_max = np.argmin(row_grad), np.argmax(row_grad)

    col_com = np.sum(np.multiply(np.arange(col_mean.shape[0]), col_mean), axis=0)/col_mean.sum()
    row_com = np.sum(np.multiply(np.arange(row_mean.shape[0]), row_mean), axis=0)/row_mean.sum()

    col_radius = (col_min - col_max)/2.
    row_radius = (row_min - row_max)/2.

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
