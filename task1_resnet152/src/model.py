import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet152_Weights

def load_resnet152(pretrained=True):
    """
    Load ResNet-152 model using locally cached ImageNet pretrained weights.
    """
    if pretrained:
        try:
            weights = ResNet152_Weights.IMAGENET1K_V1
            model = models.resnet152(weights=weights)
        except Exception:
            model = models.resnet152(pretrained=True)
    else:
        model = models.resnet152(weights=None)
    return model


def replace_head(model, num_classes=10):
    """
    Replace final fully-connected (fc) classification head.
    """
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def freeze_backbone(model):
    """
    Freeze all backbone parameters, leaving only final fc trainable.
    """
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

def unfreeze_final_block(model):
    """
    Unfreeze layer4 and final fc layer for fine-tuning.
    """
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

def unfreeze_all(model):
    """
    Unfreeze all parameters across the backbone and head.
    """
    for param in model.parameters():
        param.requires_grad = True

def unfreeze_layers(model, layer_names):
    """
    Unfreeze specific layer stages by name (e.g. ['layer4']).
    """
    for name, param in model.named_parameters():
        if any(lname in name for lname in layer_names) or 'fc' in name:
            param.requires_grad = True

