import cv2
import numpy as np
from typing import Tuple


def boundaryRectangle(img: np.array, th1: int=10) -> Tuple[list, list]:
    """Finds the rectangle that circumferences an endoscopic image.

    Args:
        img (np.array): Input image of shape HxWxC
        th1 (int): Whiten threshold, each pixel where value > th1 is whitened

    Return:
        rectangle (Tuple[list, list]): Top left corner and shape of found rectangle
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.where(img < th1, 0, 255).astype(np.uint8)
    
    col_mean = img.mean(axis=0)
    row_mean = img.mean(axis=1)

    bottom = np.min(np.nonzero(row_mean))
    top    = np.max(np.nonzero(row_mean))
    left   = np.min(np.nonzero(col_mean))
    right  = np.max(np.nonzero(col_mean))

    top_left = [top, left]
    shape = [bottom - top, right - left]

    return top_left, shape


def boundaryCircle(img: np.array, th1: int=10) -> Tuple[list, float]:
    """Find the circle that circumferences an endoscopic image. Works only with full view of the endoscopic image.

    Args:
        img (np.array): Input image of shape HxWxC
        th1 (int): Whiten threshold, each pixel where value > th1 is whitened

    Return:
        circle (Tuple[list, float]): Center and radius of found circle
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.where(img < th1, 0, 255).astype(np.uint8)

    col_mean = img.mean(axis=0)
    row_mean = img.mean(axis=1)

    col_com = np.sum(np.multiply(np.arange(col_mean.shape[0]), col_mean), axis=0)/col_mean.sum()
    row_com = np.sum(np.multiply(np.arange(row_mean.shape[0]), row_mean), axis=0)/row_mean.sum()

    col_radius = (np.max(np.nonzero(col_mean)) - np.min(np.nonzero(col_mean)))/2.
    row_radius = (np.max(np.nonzero(col_mean)) - np.min(np.nonzero(col_mean)))/2.

    radius = max(col_radius, row_radius)

    return [row_com, col_com], radius


if __name__ == '__main__':
    import os
    from copy import deepcopy
    from ransac_bounding_circle_detector import RansacBoundingCircleDetector

    prefix = os.getcwd()
    file = 'data/endo.mp4'

    vr = cv2.VideoCapture(os.path.join(prefix, file))

    bcd = RansacBoundingCircleDetector(buffer_size=10)
 
    th1 = 10

    while vr.isOpened():

        _, img = vr.read()
        if img is None:
            break

        img = cv2.resize(img, (640, 360))
        img = img[5:-5,:-5,:] # remove black bottom and top rows



        top_left, shape = boundaryRectangle(img, th1=th1)
        center0, radius0 = boundaryCircle(img, th1=th1)

    
        
        img_com = deepcopy(img)
        cv2.circle(img_com, (int(center0[1]), int(center0[0])), int(radius0), (0,255,255), 1)
        cv2.circle(img_com, (int(center0[1]), int(center0[0])), 2, (255,0,255), 4)

        cv2.rectangle(img_com, (top_left[1], top_left[0]), (top_left[1] + shape[1], top_left[0] + shape[0]), (255, 255, 0), 1)


        # ecd
        center1, radius1 = bcd.findBoundingCircle(img, th1=5, th2=100, th3=10, decay=1., fit='numeric', n_pts=100, n_iter=10)

        if radius1 is not None:
            center1, radius1 = center1.astype(np.int), int(radius1)

            img_ran = deepcopy(img)
            cv2.circle(img_ran, (center1[1], center1[0]), radius1, (0,255,255), 1)
            cv2.circle(img_ran, (center1[1], center1[0]), 2, (255,0,255), 4)

            # show output
            fps = 25
            cv2.imshow('img_com', img_com)
            cv2.imshow('img_ran', img_ran)
            # cv2.waitKey(int(1/25*1000))
            cv2.waitKey()
