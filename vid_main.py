import cv2
import os
import numpy as np

from endoscopy import ImageBuffer, ransacBoundaryCircle, boundaryRectangle

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

    # Generate image buffer
    ib = ImageBuffer(buffer_size=10)
 
    while vr.isOpened():

        _, img = vr.read()
        if img is None:
            break

        img = cv2.resize(img, (640, 360))
        img = img[5:-5,:-5,:] # remove black bottom and top rows

        # Append buffer and poll averaged binary images
        ib.appendBuffer(img)
        avg = ib.binaryAvg(th=5)

        top_left, shape = boundaryRectangle(avg, th=5)
        center, radius = ransacBoundaryCircle(avg, th=10, decay=1., fit='numeric', n_pts=100, n_iter=10)

        center, radius = center.astype(np.int), int(radius)

        cv2.rectangle(img, (top_left[1], top_left[0]), (top_left[1] + shape[1], top_left[0] + shape[0]), (255, 255, 0), 1)
        cv2.circle(img, (center[1], center[0]), radius, (0,255,255), 1)
        cv2.circle(img, (center[1], center[0]), 2, (255,0,255), 4)

        # show output
        fps = 25
        cv2.imshow('img', img)
        cv2.waitKey(int(1/25*1000))

        # # save output
        # vw.write(img)
