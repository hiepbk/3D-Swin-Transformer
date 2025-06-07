from .architectures import build_architecture
from .backbones import build_backbone
from .necks import build_neck
from .heads import build_head

__all__ = ['build_architecture', 'build_backbone', 'build_neck', 'build_head']