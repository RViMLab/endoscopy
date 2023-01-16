import cv2
import numpy as np
from kornia import image_to_tensor, tensor_to_image
from kornia.geometry import crop_and_resize

from endoscopy.bounding_circle_detector import BoundingCircleDetector
from endoscopy.utils import MODEL, max_rectangle_in_circle

if __name__ == "__main__":
    device = "cuda"

    detector = BoundingCircleDetector(
        model=MODEL.SEGMENTATION.UNET_RESNET_34, device=device
    )
    img = np.load("data/laparoscopic_view.npy")
    img = image_to_tensor(img, keepdim=False).float() / 255.0

    # detect circle
    center, radius = detector(img, N=1000, reduction=None)

    # find maximum rectangle in circle
    box = max_rectangle_in_circle(img.shape, center, radius)
    crp = crop_and_resize(img, box, [320, 480])

    # conversions
    img = tensor_to_image(img, False)
    crp = tensor_to_image(crp, False)
    center, radius = center.int().cpu().numpy(), radius.int().cpu().numpy()

    # plot
    cv2.circle(img, (center[0, 1], center[0, 0]), radius[0], (255, 255, 0), 2)

    cv2.imshow("img", img)
    cv2.imshow("crp", crp)
    cv2.waitKey()
