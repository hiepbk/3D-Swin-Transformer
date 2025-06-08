#import Regesitry
from deeplib.utils.registry import NECK_REGISTRY

def build_neck(cfg):
    """Build neck from config."""
    model_cfg = cfg.model
    if not hasattr(model_cfg, 'neck') or model_cfg.neck is None:
        return None
    neck_type = model_cfg.neck.name
    neck = NECK_REGISTRY.get(neck_type)(cfg)
    return neck

    
    

    
    
    
