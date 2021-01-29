import cv2
import os
import numpy as np

from endoscopy import RansacBoundaryCircleDetector

if __name__ == '__main__':
    prefix = os.getcwd()
    in_file = 'data/eye.tif'
    out_file = 'img/result_eye.png'

    img = cv2.imread(os.path.join(prefix, in_file))

    bcd = RansacBoundaryCircleDetector()
    center, radius = bcd.findBoundaryCircle(img, th1=15, th2=200, th3=5, decay=2., fit='numeric', n_pts=200, n_iter=10)

    if radius is not None:
        center, radius = center.astype(np.int), int(radius)
        cv2.circle(img, (center[1], center[0]), radius, (0,255,255), 2)  
        cv2.circle(img, (center[1], center[0]), 2, (255,0,255), 4)

        # show output
        cv2.imshow('img', img)
        cv2.waitKey()

        # save output
        cv2.imwrite(out_file, img)
