import torch.nn as nn
from torchvision import models

def get_model(model_name, num_classes=2):
    if model_name == 'vgg11':
        model = models.vgg11(pretrained=True)
        model.classifier[6] = nn.Linear(4096, num_classes)

    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    else:
        raise ValueError(f"Unknown model name '{model_name}'")

    return model
