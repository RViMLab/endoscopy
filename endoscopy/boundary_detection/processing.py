import numpy as np
from typing import Tuple, List


def maxRectangleInCircle(img_shape: np.ndarray, center: np.ndarray, radius: float, ratio: float=3./4.) -> Tuple[np.ndarray, tuple]:
    r"""Finds the maximum sized rectangle of given aspect ratio within a circle. The circle may be cropped within an image.
    For example see https://drive.google.com/file/d/1GMS1V415pAxdRkf2GoYLWjSFbx33cwew/view?usp=sharing.

    Args:
        img_shape (np.ndarray): Shape of image in which circle lies, HxWxC
        center (np.ndarray): Circle's center
        radius (float): Circle's radius
        ratio (float): Height/width ratio of rectanlge

    Return:
        top_left (np.ndarray): Top left corner of rectangle
        shape (tuple): Rectangle's shape
    """
    # Construct rectangle of given ratio with edges on circle
    w = 2*radius/np.sqrt(ratio**2 + 1)
    h = np.sqrt(4*radius**2 - w**2)

    d0 = center
    d1 = img_shape[:2] - center
    d1 -= 1  # shape encodes position + 1

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


def crop(img: np.ndarray, top_left: np.ndarray, shape: tuple) -> np.ndarray:
    r"""Crops and image, given the top left corner and the desired shape.

    Args:
        img (np.ndarray): Image of shape HxWxC
        top_left (np.ndarray): Top left corner
        shape (tuple): Cropped shape in (H, W)

    Return:
        img (np.ndarray): Cropped image
    """
    return img[
        top_left[0]:top_left[0] + shape[0],
        top_left[1]:top_left[1] + shape[1]
    ]


def isZoomed(img: np.ndarray, th: float=0.99) -> Tuple[bool, float]:
    r"""Determines if an image is zoomed by computing the average intensity.

    Args:
        img (np.ndarray): Binary image of shape HxW
        th (float): Threshold for zoomed image

    Return:
        is_zoomed (bool): Is zoomed if mean(img) > th
        confidence (float): Confidence measure, mean of image pixels
    """
    confidence = img.mean()/img.max()
    is_zoomed = confidence >= th
    return is_zoomed, confidence


def binaryAvg(imgs: List[np.ndarray], th: float=10.) -> np.ndarray:
    r"""Averages buffer and return binary image with cut-off threshold th.

    Args:
        imgs (List[np.ndarray]): Image buffer
        th (float): After averaging the buffer, everything below th is set to 0, else 255

    Return:
        avg (np.ndarray): Binary averaged buffer
    """
    avg = np.array(imgs)
    avg = avg.mean(axis=0)
    avg = np.where(avg < th, 0, 255).astype(np.uint8)
    return avg


def binaryVar(imgs: List[np.ndarray], th: float) -> np.ndarray:
    r"""Averages buffer and return binary image with cut-off threshold th.

    Args:
        imgs (List[np.ndarray]): Image buffer
        th (float): After computing the buffer's variance, everything below th is set to 255, else 0

    Return:
        var (np.ndarray): Binary variance
    """
    var = np.array(imgs)
    var = var.var(axis=0)
    var = np.where(var < th, 255, 0).astype(np.uint8)
    return var
