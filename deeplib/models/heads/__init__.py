from .cls_head import ClsHead
from .cls_head_3d import ClsHead3D
from .build import HEAD_REGISTRY, build_head

__all__ = ['ClsHead', 'ClsHead3D', 'HEAD_REGISTRY', 'build_head'] 