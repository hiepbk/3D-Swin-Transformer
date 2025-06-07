import torch
import torch.nn as nn
import torch.nn.functional as F
from .build import HEAD_REGISTRY

@HEAD_REGISTRY.register_module()
class ClsHead(nn.Module):
    """
    Simple classification head.
    
    Args:
        in_channels (int): Input channels
        num_classes (int): Number of classes
        dropout_ratio (float): Dropout ratio before the final classification layer
        init_cfg (dict): Initialization config dict
    """
    def __init__(self, cfg):
        super().__init__()
        self.in_channels = cfg.model.head.in_channels
        self.num_classes = cfg.model.head.num_classes
        self.dropout_ratio = cfg.model.head.dropout_ratio
        
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        
        self.fc = nn.Linear(self.in_channels, self.num_classes)
    
    def forward(self, x):
        """Forward function for training."""
        cls_score = self.fc(self.dropout(x) if self.dropout else x)
        return cls_score
    