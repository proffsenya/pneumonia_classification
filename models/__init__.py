import torch.nn as nn
from torchvision import models

def get_model(model_name, num_classes=2):
    if model_name == 'vgg11':
        model = models.vgg11(pretrained=True)
        model.classifier[6] = nn.Linear(4096, num_classes)
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Only 'vgg11' is supported.")

    return model
