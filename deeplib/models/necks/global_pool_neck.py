import torch
import torch.nn as nn
import torch.nn.functional as F

from .build import NECK_REGISTRY

@NECK_REGISTRY.register_module()
class GlobalPoolNeck(nn.Module):
    """
    Simple neck for classification that applies global pooling to features.
    
    Args:
        in_channels (list): List of input channels for each input feature map
        out_channels (int): Output channels after pooling and fusion
        pool_type (str): Type of pooling, 'avg' or 'max'
        fusion (str): How to fuse multiple feature maps: 'concat', 'sum', or 'select'
        selected_idx (int): Index of the feature map to select when fusion='select'
    """
    def __init__(self, 
                 cfg):
        super().__init__()
        
        self.in_channels = cfg.model.neck.in_channels
        self.out_channels = cfg.model.neck.out_channels
        self.pool_type = cfg.model.neck.pool_type
        self.fusion = cfg.model.neck.fusion
        self.selected_idx = cfg.model.neck.selected_idx
        
        # Create projection layers if needed
        if self.fusion == 'concat':
            self.projection = nn.Linear(sum(self.in_channels), self.out_channels)
        elif self.fusion == 'sum':
            self.projections = nn.ModuleList([
                nn.Conv2d(c, self.out_channels, kernel_size=1) 
                for c in self.in_channels
            ])
        elif self.fusion == 'select':
            if self.selected_idx < 0:
                self.selected_idx = len(self.in_channels) + self.selected_idx
            assert 0 <= self.selected_idx < len(self.in_channels), "Selected index out of range"
            if self.in_channels[self.selected_idx] != self.out_channels:
                self.projection = nn.Linear(self.in_channels[self.selected_idx], self.out_channels)
            else:
                self.projection = nn.Identity()
        else:
            raise ValueError(f"Unsupported fusion mode: {self.fusion}")
    
    def forward(self, inputs):
        """
        Args:
            inputs (tuple): Tuple of feature maps from backbone
            
        Returns:
            torch.Tensor: Processed features for classification
        """
        assert isinstance(inputs, (list, tuple))
        assert len(inputs) == len(self.in_channels)
        
        # Apply pooling to each feature map
        if self.pool_type == 'avg':
            pooled_features = [F.adaptive_avg_pool2d(x, 1).flatten(1) for x in inputs]
        elif self.pool_type == 'max':
            pooled_features = [F.adaptive_max_pool2d(x, 1).flatten(1) for x in inputs]
        else:
            raise ValueError(f"Unsupported pooling type: {self.pool_type}")
        
        # Fuse features according to the fusion mode
        if self.fusion == 'concat':
            fused = torch.cat(pooled_features, dim=1)
            output = self.projection(fused)
        elif self.fusion == 'sum':
            projected = [proj(feat.view(feat.size(0), feat.size(1), 1, 1)) 
                         for proj, feat in zip(self.projections, pooled_features)]
            pooled_projected = [F.adaptive_avg_pool2d(x, 1).flatten(1) for x in projected]
            output = sum(pooled_projected)
        elif self.fusion == 'select':
            output = self.projection(pooled_features[self.selected_idx])
        
        return output 