import torch
from enum import Enum


class SEGMENTATION_MODEL(Enum):
    UNET_RESNET_34 = "https://github.com/RViMLab/endoscopy/releases/download/0.0.1/seg_unet_resnet_34.pt"
    UNET_RESNET_34_TINY = "https://github.com/RViMLab/endoscopy/releases/download/0.0.1/seg_unet_resnet_34_tiny.pt"


def load_model(device: str="cuda", model_enum: SEGMENTATION_MODEL=SEGMENTATION_MODEL.UNET_RESNET_34):
    model = torch.hub.load_state_dict_from_url(model_enum.value)
    return model.eval().to(device)
