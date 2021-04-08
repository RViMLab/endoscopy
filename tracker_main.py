import cv2
import os
import numpy as np

from endoscopy import CoMBoundaryTracker, bilateralSegmentation


if __name__ == '__main__':
    prefix = os.getcwd()
    in_file = 'data/eye.tif'

    img = cv2.imread(os.path.join(prefix, in_file))
    mask = bilateralSegmentation(img, 0.2)

    tracker = CoMBoundaryTracker()
    circle, rectangle = tracker.circle, tracker.rectangle

    print('Initial Values')
    print('Circle: {}\{}'.format(circle[0], circle[1]))
    print('Rectanlge: {}\{}'.format(rectangle[0], rectangle[1]))

    print('Updating tracker...')
    circle = tracker.updateBoundaryCircle(mask)
    rectangle = tracker.updateBoundaryRectangle(mask)
    print('Done.')

    print('Updated Values')
    print('Circle: {}\{}'.format(circle[0], circle[1]))
    print('Rectanlge: {}\{}'.format(rectangle[0], rectangle[1]))

    # draw results
    center, radius = circle
    top_left, shape = rectangle
    cv2.circle(img, (center[1], center[0]), radius, (255, 255, 0))
    cv2.rectangle(img, (top_left[1], top_left[0]), (top_left[1]+shape[1], top_left[0]+shape[0]), (255, 0, 255))

    cv2.imshow('img', img)
    cv2.waitKey()
