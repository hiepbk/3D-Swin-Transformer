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
        depth = head_cfg.depth
        mlp_blocks = []
        for _ in range(depth):
            mlp_blocks.append(MlpBasicBlock(in_channels, in_channels, dropout))
        mlp_blocks.append(nn.Linear(in_channels, num_classes))
        self.mlp_blocks = nn.Sequential(*mlp_blocks)

    def forward(self, x):
        return self.mlp_blocks(x)
    

class MlpBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, out_channels),
        )
        
    def forward(self, x):
        return self.mlp(x)