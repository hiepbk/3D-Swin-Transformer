import torch
import torch.nn as nn
import torchvision.models as models

# Import registry after it's defined
from .build import BACKBONE_REGISTRY

@BACKBONE_REGISTRY.register_module()
class ResNet(nn.Module):    
    """
    ResNet backbone.
    """
    def __init__(self, cfg):
        super().__init__()
        self.depth = cfg.model.backbone.depth
        self.weights = cfg.model.backbone.weights
        self.frozen_stages = cfg.model.backbone.frozen_stages
        self.out_indices = cfg.model.backbone.out_indices

        if self.depth == 18:
            self.resnet = models.resnet18(weights=self.weights)
        elif self.depth == 34:
            self.resnet = models.resnet34(weights=self.weights)
        elif self.depth == 50:
            self.resnet = models.resnet50(weights=self.weights)
        elif self.depth == 101:
            self.resnet = models.resnet101(weights=self.weights)
        elif self.depth == 152:
            self.resnet = models.resnet152(weights=self.weights)
        else:
            raise ValueError(f"Invalid depth {self.depth} for ResNet")
        
        self.out_indices = self.out_indices
        self.frozen_stages = self.frozen_stages
        
        # Freeze stages
        if self.frozen_stages >= 0:
            self.resnet.conv1.eval()
            self.resnet.bn1.eval()
            for param in self.resnet.conv1.parameters():
                param.requires_grad = False
            for param in self.resnet.bn1.parameters():
                param.requires_grad = False
        
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self.resnet, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        outputs = []
        for i in range(1, 5):
            layer = getattr(self.resnet, f'layer{i}')
            x = layer(x)
            if i - 1 in self.out_indices:
                outputs.append(x)
        
        return tuple(outputs) 