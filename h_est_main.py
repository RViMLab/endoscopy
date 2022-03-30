import cv2
import numpy as np
from kornia import tensor_to_image, image_to_tensor
from kornia.geometry import warp_perspective

from endoscopy.homography_estimator import HomographyEstimator
from endoscopy.utils import MODEL, yt_alpha_blend


if __name__ == "__main__":
    device = "cuda"

    homography_estimator = HomographyEstimator(model=MODEL.HOMOGRAPHY_ESTIMATION.RESNET_34, device=device)
    img = np.load("data/laparoscopic_view.npy")
    img = image_to_tensor(img, keepdim=False).float()/255.
 
    # estimate homography
    h, duv = homography_estimator(img, img)

    # visualize through blend
    wrp = warp_perspective(img, h.inverse(), img.shape[-2:])
    blend = yt_alpha_blend(img, wrp)

    blend = tensor_to_image(blend.cpu(), keepdim=False)
    cv2.imshow("blend", blend)
    cv2.waitKey()
