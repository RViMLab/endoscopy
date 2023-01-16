import cv2
import torch
from kornia import image_to_tensor, tensor_to_image
from kornia.geometry import crop_and_resize

from endoscopy.bounding_circle_detector import BoundingCircleDetector
from endoscopy.utils import MODEL, max_rectangle_in_circle

if __name__ == "__main__":
    device = "cuda"

    detector = BoundingCircleDetector(
        model=MODEL.SEGMENTATION.UNET_RESNET_34, device=device
    )
    vc = cv2.VideoCapture("data/endo.mp4")
    B = 5
    buffer = []

    while vc.isOpened():
        _, img = vc.read()
        if img is None:
            break
        buffer.append(image_to_tensor(img, keepdim=True).float() / 255.0)
        if len(buffer) >= B:
            imgs = torch.stack(buffer)

            # detect circle
            center, radius = detector(imgs, N=1000, reduction=None)

            # find maximum rectangle in circle
            box = max_rectangle_in_circle(imgs.shape, center, radius)
            crps = crop_and_resize(imgs, box, [320, 480])

            # conversions
            img = tensor_to_image(imgs[0], False).copy()
            crp = tensor_to_image(crps[0], False).copy()
            center, radius = center.int().cpu().numpy(), radius.int().cpu().numpy()

            # plot
            cv2.circle(img, (center[0, 1], center[0, 0]), radius[0], (255, 255, 0), 2)

            cv2.imshow("img", img)
            cv2.imshow("crp", crp)
            cv2.waitKey()

            buffer.clear()
