import torch

model_urls = {
    "model": "https://github.com/RViMLab/endoscopy/releases/download/0.0.1/model.pt"
}

def load_model(device: str="cuda", name: str="model"):
    model = torch.hub.load_state_dict_from_url(model_urls[name])
    return model.eval().to(device)
