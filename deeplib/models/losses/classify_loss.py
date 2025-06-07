import torch
import torch.nn as nn
import torch.nn.functional as F

from .build import LOSS_REGISTRY

@LOSS_REGISTRY.register_module()
class FocalLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.dataset.label_mode == 'multi', "FocalLoss only supports multi-label classification"
        
        self.alpha = cfg.model.loss.alpha
        self.gamma = cfg.model.loss.gamma
        
    def forward(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target)
    
@LOSS_REGISTRY.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.dataset.label_mode == 'single', "CrossEntropyLoss only supports single-label classification"
        self.loss_name = cfg.model.loss.name
        self.loss_weight = cfg.model.loss.weight
        self.loss_func = nn.CrossEntropyLoss()
        
    def forward(self, pred, target):
        return self.loss_func(pred, target)

