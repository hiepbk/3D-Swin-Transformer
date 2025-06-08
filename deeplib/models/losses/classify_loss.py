import torch
import torch.nn as nn
import torch.nn.functional as F
from deeplib.utils.registry import LOSS_REGISTRY

class FocalLossBase(nn.Module):
    """Base class for Focal Loss implementation"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        raise NotImplementedError

class FocalLossSingle(FocalLossBase):
    """Focal Loss for single-label classification"""
    def forward(self, pred, target):
        log_softmax = F.log_softmax(pred, dim=1)
        softmax = torch.exp(log_softmax)
        
        # Get probability of target class
        pt = softmax.gather(1, target.unsqueeze(1)).squeeze(1)
        
        # Calculate focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Fix: handle both float and tensor alpha
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_weight = self.alpha
            else:
                # assume tensor or list
                alpha_tensor = self.alpha
                if not torch.is_tensor(alpha_tensor):
                    alpha_tensor = torch.tensor(alpha_tensor, device=pred.device, dtype=pred.dtype)
                alpha_weight = alpha_tensor[target]
            focal_weight = alpha_weight * focal_weight
        
        # Calculate loss
        loss = F.nll_loss(log_softmax, target, reduction='none')
        weighted_loss = focal_weight * loss
        
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        return weighted_loss

class FocalLossMulti(FocalLossBase):
    """Focal Loss for multi-label classification"""
    def forward(self, pred, target):
        sigmoid_pred = torch.sigmoid(pred)
        
        # Calculate probability of target class
        pt = torch.where(target == 1, sigmoid_pred, 1 - sigmoid_pred)
        
        # Calculate focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            alpha_weight = torch.where(target == 1, self.alpha, 1 - self.alpha)
            focal_weight = alpha_weight * focal_weight
        
        # Calculate loss
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        weighted_loss = focal_weight * loss
        
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        return weighted_loss

@LOSS_REGISTRY.register_module()
class FocalLoss(nn.Module):
    def __init__(self, loss_cfg):
        super().__init__()
        self.loss_name = loss_cfg.name
        self.label_mode = loss_cfg.label_mode
        self.loss_weight = loss_cfg.loss_weight
        alpha = loss_cfg.alpha if hasattr(loss_cfg, 'alpha') else None
        gamma = loss_cfg.gamma if hasattr(loss_cfg, 'gamma') else 2.0
        
        if self.label_mode == 'multi':
            self.loss_fn = FocalLossMulti(alpha=alpha, gamma=gamma)
        else:
            self.loss_fn = FocalLossSingle(alpha=alpha, gamma=gamma)
    
    def forward(self, pred, target):
        # print(f"pred.shape: {pred.shape}, target.shape: {target.shape}, in focal loss")
        # print(f"pred: {pred}, target: {target}")
        return self.loss_fn(pred, target)
    
@LOSS_REGISTRY.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(self, loss_cfg):
        super().__init__()
        self.loss_name = loss_cfg.name
        self.loss_weight = loss_cfg.loss_weight
        self.label_mode = loss_cfg.label_mode
        
        # Get class weights if provided
        if hasattr(loss_cfg, 'class_weight'):
            self.class_weight = torch.tensor(loss_cfg.class_weight, device="cuda")
        else:
            self.class_weight = None
            
        if self.label_mode == 'multi':
            self.loss_func = nn.BCEWithLogitsLoss(pos_weight=self.class_weight)
        else:
            self.loss_func = nn.CrossEntropyLoss(weight=self.class_weight)
        
    def forward(self, pred, target):
        # print(f"pred.shape: {pred.shape}, target.shape: {target.shape}, in cross entropy loss")
        # print(f"pred: {pred}, target: {target}")
        return self.loss_func(pred, target)

# This is wrapper for all loss functions, and also return the total loss
@LOSS_REGISTRY.register_module()
class ClassifyLoss(nn.Module):
    def __init__(self, loss_dict):
        super().__init__()
        self.loss_dict = loss_dict
        
    def forward(self, cls_score, label):
        """Forward function.
        Args:
            cls_score (torch.Tensor): The classification score.
            label (torch.Tensor): The ground truth label.
        Returns:
            dict: A dictionary containing individual losses and total loss.
        """
        losses = dict()
        # Create a copy of loss_dict to avoid modification during iteration
        loss_dict = dict(self.loss_dict)
        total_loss = 0
        for loss_name, loss_func in loss_dict.items():
            if loss_name == 'CrossEntropyLoss':
                losses[loss_name] = loss_func(cls_score, label)
            elif loss_name == 'FocalLoss':
                losses[loss_name] = loss_func(cls_score, label)
            else:
                raise NotImplementedError(f'Loss {loss_name} is not implemented.')
            total_loss += losses[loss_name]
        
        losses['Loss'] = total_loss
        return losses

