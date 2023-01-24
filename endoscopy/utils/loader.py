from enum import Enum

import torch


class MODEL(object):
    r"""Model meta class.

    URLs to models are to include a 'cpu' or 'cuda' tag, respectively.
    """

    class HOMOGRAPHY_ESTIMATION_ENUM(Enum):
        H_48_RESNET_34 = "https://github.com/RViMLab/endoscopy/releases/download/0.1.1/h_est_48_resnet_34_{}.pt"
        H_64_RESNET_34 = "https://github.com/RViMLab/endoscopy/releases/download/0.1.1/h_est_64_resnet_34_{}.pt"

    class HOMOGRAPHY_PREDICTION_ENUM(Enum):
        RESNET_34_IN_27 = "https://github.com/RViMLab/endoscopy/releases/download/0.1.1/h_pred_resnet_34_in_channels_27_{}.pt"
        RESNET_50_IN_27 = "https://github.com/RViMLab/endoscopy/releases/download/0.1.1/h_pred_resnet_50_in_channels_27_{}.pt_"

    class SEGMENTATION_ENUM(Enum):
        UNET_RESNET_34 = "https://github.com/RViMLab/endoscopy/releases/download/0.1.1/seg_unet_resnet_34_{}.pt"
        UNET_RESNET_34_TINY = "https://github.com/RViMLab/endoscopy/releases/download/0.1.1/seg_unet_resnet_34_tiny_{}.pt"

    HOMOGRAPHY_ESTIMATION = HOMOGRAPHY_ESTIMATION_ENUM
    HOMOGRAPHY_PREDICTION = HOMOGRAPHY_PREDICTION_ENUM
    SEGMENTATION = SEGMENTATION_ENUM


def load_model(model: MODEL, device: str = "cuda"):
    model = torch.hub.load_state_dict_from_url(model.value.format(device))
    return model.eval()
