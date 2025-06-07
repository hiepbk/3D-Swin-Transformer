from .build import build_loss, LOSS_REGISTRY
from .classify_loss import FocalLoss, CrossEntropyLoss

__all__ = ['build_loss', 'LOSS_REGISTRY', 'FocalLoss', 'CrossEntropyLoss']