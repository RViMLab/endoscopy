import torch
from enum import Enum


class MODEL(object):
    r"""Model meta class.

    URLs to models are to include a 'cpu' or 'cuda' tag, respectively.
    """
    class HOMOGRAPHY_ESTIMATION_ENUM(Enum):
        RESNET_34 = "https://github.com/RViMLab/endoscopy/releases/download/0.1.1/h_est_resnet_34_{}.pt"

    class SEGMENTATION_ENUM(Enum):
        UNET_RESNET_34 = "https://github.com/RViMLab/endoscopy/releases/download/0.1.1/seg_unet_resnet_34_{}.pt"
        UNET_RESNET_34_TINY = "https://github.com/RViMLab/endoscopy/releases/download/0.1.1/seg_unet_resnet_34_{}_tiny.pt"

    HOMOGRAPHY_ESTIMATION = HOMOGRAPHY_ESTIMATION_ENUM
    SEGMENTATION = SEGMENTATION_ENUM


def load_model(model: MODEL, device: str="cuda"):
    model = torch.hub.load_state_dict_from_url(model.value.format(device))
    return model.eval()
