import cv2
import numpy as np
from kornia import image_to_tensor, tensor_to_image
from kornia.geometry import warp_perspective

from endoscopy.homography_estimator import HomographyEstimator
from endoscopy.utils import MODEL, yt_alpha_blend


def main() -> None:
    device = "cuda"

    homography_estimator = HomographyEstimator(
        model=MODEL.HOMOGRAPHY_ESTIMATION.H_48_RESNET_34, device=device
    )
    img = np.load("data/cropped_sample_sequence/frame_200.npy")
    wrp = np.load("data/cropped_sample_sequence/frame_209.npy")
    img = image_to_tensor(img, keepdim=False).float() / 255.0
    wrp = image_to_tensor(wrp, keepdim=False).float() / 255.0

    # estimate homography
    h, duv = homography_estimator(img, wrp)

    # visualize through blend
    wrp_est = warp_perspective(img, h.inverse(), img.shape[-2:])
    blend = yt_alpha_blend(wrp_est, wrp)

    blend = tensor_to_image(blend.cpu(), keepdim=False)
    cv2.imshow("blend", blend)
    cv2.waitKey()


if __name__ == "__main__":
    main()
