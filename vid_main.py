import cv2
import numpy as np
from kornia import tensor_to_image, image_to_tensor
from kornia.geometry import crop_and_resize

from endoscopy.bounding_circle_detector import BoundingCircleDetector
from endoscopy.utils import max_rectangle_in_circle, SEGMENTATION_MODEL


if __name__ == "__main__":
    device = "cuda"

    detector = BoundingCircleDetector(device=device, model_enum=SEGMENTATION_MODEL.UNET_RESNET_34)
    vc = cv2.VideoCapture("data/endo.mp4")

    while vc.isOpened():
        _, img = vc.read()
        if img is None:
            break

        img = image_to_tensor(img, keepdim=False).float()/255.

        # detect circle
        center, radius = detector(img, N=10000, reduction=None)

        # find maximum rectangle in circle
        box = max_rectangle_in_circle(img.shape, center, radius)
        crp = crop_and_resize(img, box, [320, 480])

        # conversions
        img = tensor_to_image(img[0], False).copy()
        crp = tensor_to_image(crp[0], False).copy()
        center, radius = center.int().cpu().numpy(), radius.int().cpu().numpy()

        # plot
        cv2.circle(img, (center[0, 1], center[0, 0]), radius[0], (255, 255, 0), 2)

        cv2.imshow("img", img)
        cv2.imshow("crp", crp)
        cv2.waitKey()
