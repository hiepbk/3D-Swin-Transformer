import torch
import torch.nn as nn
import torch.nn.functional as F
from .build import HEAD_REGISTRY

@HEAD_REGISTRY.register_module()
class ClsHead3D(nn.Module):
    """Linear classification head for Swin Transformer.
    
    Args:
        cfg: Configuration object containing head parameters
    """
    def __init__(self, cfg):
        super().__init__()
        head_cfg = cfg.model.head
        num_classes = head_cfg.num_classes
        in_channels = head_cfg.in_channels
        dropout = head_cfg.dropout
        
        self.head = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, num_classes)
        )
        
    def forward(self, x):
        return self.head(x) 