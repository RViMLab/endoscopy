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
    n_images = 10
    imgs = load_images(
        path="data/cropped_sample_sequence",
        device=device,
        increment=5,
        n_images=n_images,
    )

    homography_predictor = HomographyPredictor(
        predictor=MODEL.HOMOGRAPHY_PREDICTION.RESNET_34_IN_27,
        device=device,
    )

    h, duv = homography_predictor(imgs[:, : n_images - 1])
    print(duv)

    wrps_pred = warp_perspective(imgs[:, 0], h[:].inverse(), imgs.shape[-2:])
    blends = yt_alpha_blend(wrps_pred, imgs[:, -1])
    blends_no_pred = yt_alpha_blend(imgs[:, 0], imgs[:, -1])

    import matplotlib.pyplot as plt

    for blend, blend_no_pred in zip(blends, blends_no_pred):
        plot = np.concatenate(
            [
                tensor_to_image(blend, keepdim=False),
                tensor_to_image(blend_no_pred, keepdim=False),
            ],
            axis=1,
        )
        plt.imshow(plot)
        plt.show()


if __name__ == "__main__":
    main()
