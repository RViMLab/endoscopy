import numpy as np
from typing import Tuple


def maxRectangleInCircle(img_shape: np.array, center: np.array, radius: float, ratio: float=3./4.) -> Tuple[np.array, tuple]:
    """Finds the maximum sized rectangle of given aspect ratio within a circle. The circle may be cropped within an image.
    For example see https://drive.google.com/file/d/1GMS1V415pAxdRkf2GoYLWjSFbx33cwew/view?usp=sharing.

    Args:
        img_shape (np.array): Shape of image in which circle lies, HxWxC
        center (np.array): Circle's center
        radius (float): Circle's radius
        ratio (float): Height/width ratio of rectanlge

    Return:
        top_left (np.array): Top left corner of rectangle
        shape (tuple): Rectangle's shape
    """
    # Construct rectangle of given ratio with edges on circle
    w = 2*radius/np.sqrt(ratio**2 + 1)
    h = np.sqrt(4*radius**2 - w**2)

    d0 = center
    d1 = img_shape[:2] - center
    d1 -= 1  # shape encode position + 1

    new_ratio_shape = (
        min(2*min(d0[0], d1[0]) + 1, h),
        min(2*min(d0[1], d1[1]) + 1, w)
    )

    new_ratio = new_ratio_shape[0]/new_ratio_shape[1]

    if ratio/new_ratio < 1:    # update ratio preserving height
        shape = (ratio*new_ratio_shape[1], new_ratio_shape[1])
    elif ratio/new_ratio > 1:  # update ratio preserving width
        shape = (new_ratio_shape[0], new_ratio_shape[0]/ratio)
    else:
        shape = new_ratio_shape

    top_left = np.array([
        center[0] - (shape[0] - 1)/2, center[1] - (shape[1] - 1)/2
    ])

    return top_left, shape


def isZoomed(img: np.array, th: float=0.99) -> Tuple[bool, float]:
    """Determines if an image is zoomed by computing the average intensity.

    Args:
        img (np.array): Binary image of shape HxW
        th (float): Threshold for zoomed image

    Return:
        is_zoomed (bool): Is zoomed if mean(img) > th
        confidence (float): Confidence measure, mean of image pixels
    """
    confidence = img.mean()/img.max()
    is_zoomed = confidence >= th
    return is_zoomed, confidence
