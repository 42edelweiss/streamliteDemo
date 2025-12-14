import torch
import torch.nn as nn
from torchvision import models

class MobileNetV3Classifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, version='small'):
        super(MobileNetV3Classifier, self).__init__()
        
        if version == 'small':
            self.mobilenet = models.mobilenet_v3_small(weights='DEFAULT' if pretrained else None)
        elif version == 'large':
            self.mobilenet = models.mobilenet_v3_large(weights='DEFAULT' if pretrained else None)
        else:
            raise ValueError('version must be small or large')
        
        in_features = self.mobilenet.classifier[-1].in_features
        self.mobilenet.classifier[-1] = nn.Linear(in_features, num_classes)
        self.version = version
    
    def forward(self, x):
        return self.mobilenet(x)
    
    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'version': self.version,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'total_params_M': total_params / 1e6
        }