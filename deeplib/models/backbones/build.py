#import Regesitry
from deeplib.utils.registry import Registry

# Create registry
BACKBONE_REGISTRY = Registry('backbone')

def build_backbone(cfg):
    """Build backbone from config."""
    backbone_type = cfg.model.backbone.name
    backbone = BACKBONE_REGISTRY.get(backbone_type)(cfg)
    return backbone

    
    
    
