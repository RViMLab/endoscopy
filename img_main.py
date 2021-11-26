import cv2
import numpy as np
from kornia import tensor_to_image, image_to_tensor

from endoscopy.bounding_circle_detector import BoundingCircleDetector


if __name__ == "__main__":
    device = "cuda"

    detector = BoundingCircleDetector(device=device)
    img = np.load("data/laparoscopic_view.npy")
    img = image_to_tensor(img, keepdim=False).float()/255.

    center, radius = detector(img)

    img = tensor_to_image(img, False)

    # conversions
    center, radius = center.cpu().numpy().astype(int), int(radius)

    # plot
    cv2.circle(img, (center[1,0], center[0,0]), radius, (255, 255, 0), 2)

    cv2.imshow("img", img)
    cv2.waitKey()
