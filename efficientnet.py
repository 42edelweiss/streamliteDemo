import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(EfficientNetClassifier, self).__init__()
        
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.efficientnet = efficientnet_b0(weights=weights)
        
        in_features = self.efficientnet.classifier[-1].in_features
        self.efficientnet.classifier[-1] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.efficientnet(x)
    
    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'model': 'EfficientNet-B0',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'total_params_M': total_params / 1e6
        }

