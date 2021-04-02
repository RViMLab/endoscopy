import cv2
import os
import numpy as np

from endoscopy import ImageBuffer, ransacBoundaryCircle, maxRectangleInCircle

if __name__ == '__main__':
    prefix = os.getcwd()
    in_file = 'data/endo.mp4'
    out_file = 'data/result_endo.avi'

    offset = 5

    vr = cv2.VideoCapture(os.path.join(prefix, in_file))
    vw = cv2.VideoWriter(
        os.path.join(prefix, out_file), 
        cv2.VideoWriter_fourcc('M','J','P','G'),
        25,
        (int(vr.get(cv2.CAP_PROP_FRAME_WIDTH) - 2*offset), int(vr.get(cv2.CAP_PROP_FRAME_HEIGHT) - 2*offset))
    )

    # Generate image buffer
    ib = ImageBuffer(buffer_size=10)
 
    while vr.isOpened():

        _, img = vr.read()
        if img is None:
            break

        img = img[offset:-offset,offset:-offset,:] # remove black bottom and top rows

        # Append buffer and poll averaged binary images
        ib.appendBuffer(img)
        avg = ib.binaryAvg(th=5)

        center, radius = ransacBoundaryCircle(avg, th=10, decay=1., fit='numeric', n_pts=100, n_iter=10)
        top_left, shape = maxRectangleInCircle(avg.shape, center, radius)

        top_left, shape = top_left.astype(np.int), tuple(map(int, shape))
        center, radius = center.astype(np.int), int(radius)

        cv2.rectangle(img, (top_left[1], top_left[0]), (top_left[1] + shape[1], top_left[0] + shape[0]), (255, 255, 0), 2)
        cv2.circle(img, (center[1], center[0]), radius, (0,255,255), 2)
        cv2.circle(img, (center[1], center[0]), 2, (0,255,255), 2)

        # show output
        fps = 25
        cv2.imshow('avg', avg)
        cv2.imshow('img', img)
        cv2.waitKey(int(1/25*1000))

        # # save output
        # vw.write(img)
