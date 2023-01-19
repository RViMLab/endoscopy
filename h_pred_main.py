import os
import re
from typing import List, Union

import cv2
import numpy as np
import torch
from kornia import tensor_to_image
from kornia.geometry import warp_perspective

from endoscopy.homography_predictor import HomographyPredictor
from endoscopy.utils import MODEL, yt_alpha_blend


def atoi(text: str) -> Union[int, str]:
    return int(text) if text.isdigit() else text


def natural_keys(text: str) -> List[int]:
    r"""Sorts in human order, see
        https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside

    Example:
        sorted_list = sorted(list, key=natural_keys)
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def load_images(
    path: str, device: str, increment: int = 1, n_images: int = None
) -> torch.Tensor:
    file_names = [file_name for file_name in os.listdir(path)]
    file_names = sorted(file_names, key=natural_keys)
    file_names = file_names[::increment]

    images = []
    for file_name in file_names:
        img = np.load(os.path.join(path, file_name))
        img = cv2.resize(img, (320, 240))
        images.append(img)
        if n_images:
            if len(images) >= n_images:
                break
    return (
        torch.tensor(images, device=device).permute(0, 3, 1, 2).unsqueeze(0).float()
        / 255.0
    )


def main() -> None:
    device = "cuda"
    imgs = load_images("data/cropped_sample_sequence", device, 5, None)

    homography_predictor = HomographyPredictor(
        estimator=MODEL.HOMOGRAPHY_ESTIMATION.H_48_RESNET_34,
        predictor=MODEL.HOMOGRAPHY_PREDICTION.H_64_FEATURE_LSTM,
        device=device,
    )

    hx = None
    for _ in range(2):
        hs_ip1, duvs_ip1, hx = homography_predictor(imgs, hx)

    wrps_pred = warp_perspective(
        imgs[0, 2:-1], hs_ip1[0, :-1].inverse(), imgs.shape[-2:]
    )
    blends = yt_alpha_blend(wrps_pred, imgs[0, 3:])
    for blend in blends:
        cv2.imshow("blend", tensor_to_image(blend, keepdim=False))
        cv2.waitKey()


if __name__ == "__main__":
    main()
