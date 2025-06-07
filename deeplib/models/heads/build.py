#import Regesitry
from deeplib.utils.registry import Registry

HEAD_REGISTRY = Registry('head')

def build_head(cfg):
    """Build head from config."""
    head_type = cfg.model.head.name
    head = HEAD_REGISTRY.get(head_type)(cfg)
    return head

    
    
    
