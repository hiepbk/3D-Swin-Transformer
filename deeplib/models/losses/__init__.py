from .build import build_loss
from .classify_loss import FocalLoss, CrossEntropyLoss, ClassifyLoss

__all__ = ['build_loss', 'FocalLoss', 'CrossEntropyLoss', 'ClassifyLoss']