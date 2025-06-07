# First import the registry and build function
from .build import BACKBONE_REGISTRY, build_backbone

# Then import the models that use the registry
from .resnet import ResNet
from .swin_transformer_3d import SwinTransformer3D

__all__ = ['BACKBONE_REGISTRY', 'build_backbone', 'ResNet', 'SwinTransformer3D']

