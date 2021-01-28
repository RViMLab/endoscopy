import cv2
import os
import numpy as np

from endoscopy_bounding_circle_detector import EndoscopyBoundingCircleDetector

if __name__ == '__main__':
    prefix = os.getcwd()
    in_file = 'data/endo.mp4'
    out_file = 'data/result_endo.avi'

    vr = cv2.VideoCapture(os.path.join(prefix, in_file))
    vw = cv2.VideoWriter(
        os.path.join(prefix, out_file), 
        cv2.VideoWriter_fourcc('M','J','P','G'),
        25,
        (int(vr.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vr.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )

    ebcd = EndoscopyBoundingCircleDetector(buffer_size=1)
 
    while vr.isOpened():

        _, img = vr.read()
        if img is None:
            break

        center, radius = ebcd.findBoundingCircle(img, th1=5, th2=200, th3=10., decay=2., fit='analytic', n_pts=100, n_iter=4)

        if radius is not None:
            center, radius = center.astype(np.int), int(radius)

            cv2.circle(img, (center[1], center[0]), radius, (0,255,255), 1)
            cv2.circle(img, (center[1], center[0]), 2, (255,0,255), 4)

            # show output
            fps = 25
            cv2.imshow('img', img)
            cv2.waitKey(int(1/25*1000))

            # # save output
            # vw.write(img)
