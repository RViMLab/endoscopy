import torch

model_urls = {
    "segmentation_unet_efficient_net_b4": "https://github.com/RViMLab/endoscopy/releases/download/0.0.1/segmentation_unet_efficient_net_b4.ckpt",
    "segmentation_unet_resnet_34": "https://github.com/RViMLab/endoscopy/releases/download/0.0.1/segmentation_unet_resnet_34.ckpt",
    "segmentation_unet_resnet_34_tiny": "https://github.com/RViMLab/endoscopy/releases/download/0.0.1/segmentation_unet_resnet_34_tiny.ckpt"
}

def load_model(device: str="cuda", name: str="segmentation_unet_resnet_34_tiny"):
    model = torch.hub.load_state_dict_from_url(model_urls[name])
    return model.eval().to(device)
