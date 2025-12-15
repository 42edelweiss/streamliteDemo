import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_resnet(num_classes=2, pretrained=True):
    if pretrained:
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
    else:
        model = resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def set_trainable(model, mode: str):
    """
    mode:
      - "all": train everything
      - "head": train only fc
      - "layer4+head": train layer4 and fc
      - "layer3+layer4+head": train layer3, layer4, fc
    """
    for p in model.parameters():
        p.requires_grad = False

    if mode == "all":
        for p in model.parameters():
            p.requires_grad = True
        return

    for p in model.fc.parameters():
        p.requires_grad = True

    if mode in ("layer4+head", "layer3+layer4+head"):
        for p in model.layer4.parameters():
            p.requires_grad = True

    if mode == "layer3+layer4+head":
        for p in model.layer3.parameters():
            p.requires_grad = True
