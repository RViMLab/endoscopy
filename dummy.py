import cv2
import numpy as np
from pathlib import Path
from kornia import tensor_to_image, image_to_tensor

from endoscopy.bounding_circle_detector import BoundingCircleDetector

import time

if __name__ == "__main__":
    detector = BoundingCircleDetector()

    
    # path = Path("/media/martin/Samsung_T5/data/endoscopic_data/fov_segmentation/frame")

    # n = 0
    # dt = 0
    # for file in path.iterdir():
    img = np.load("data/laparoscopic_view.npy")
    img = image_to_tensor(img, keepdim=False).float()/255.

    seg = detector(img)

    seg = tensor_to_image(seg, keepdim=False)
    cv2.imshow("seg", seg)
    cv2.waitKey()
