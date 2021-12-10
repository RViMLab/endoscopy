import torch

model_urls = {
    "seg_unet_resnet_34": "https://github.com/RViMLab/endoscopy/releases/download/0.0.1/seg_unet_resnet_34.pt",
    "seg_unet_resnet_34_tiny": "https://github.com/RViMLab/endoscopy/releases/download/0.0.1/seg_unet_resnet_34_tiny.pt"
}

def load_model(device: str="cuda", name: str="segmentation_unet_resnet_34_tiny"):
    model = torch.hub.load_state_dict_from_url(model_urls[name])
    return model.eval().to(device)
