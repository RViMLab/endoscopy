import cv2
import numpy as np
from kornia import image_to_tensor, tensor_to_image
from kornia.geometry import warp_perspective

from endoscopy.homography_predictor import HomographyPredictor
from endoscopy.utils import MODEL, yt_alpha_blend


def main() -> None:
    device = "cuda"

    homography_predictor = HomographyPredictor(
        model=MODEL.HOMOGRAPHY_PREDICTION.H_FEATURE_LSTM, device=device
    )



    homography_predictor()


if __name__ == "__main__":
    main()
