# Define the classification metric

import numpy as np
import torch
from collections import defaultdict

class ClassificationMetric:
    """
    Class to accumulate and calculate classification metrics.
    Supports both multi-class and multi-label scenarios.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.correct = 0
        self.total = 0
        self.true_positives = defaultdict(int)
        self.false_positives = defaultdict(int)
        self.false_negatives = defaultdict(int)
        self.accumulated_loss = 0
        self.num_batches = 0
        
    def update(self, pred, target, loss=None):
        """
        Update metrics with new predictions
        
        Args:
            pred (torch.Tensor): Model predictions (N, C) or (N, num_classes)
            target (torch.Tensor): Ground truth labels (N,) or (N, num_classes)
            loss (float, optional): Batch loss value
        """
        if target.dim() > 1:  # Multi-label case
            self._update_multilabel(pred, target)
        else:  # Multi-class case
            self._update_multiclass(pred, target)
            
        if loss is not None:
            self.accumulated_loss += loss
            self.num_batches += 1
    
    def _update_multiclass(self, pred, target):
        """Update metrics for multi-class scenario"""
        pred_classes = pred.argmax(dim=1)
        self.correct += (pred_classes == target).sum().item()
        self.total += target.size(0)
        
        # Calculate per-class metrics
        for class_idx in range(pred.size(1)):
            pred_mask = pred_classes == class_idx
            target_mask = target == class_idx
            
            self.true_positives[class_idx] += (pred_mask & target_mask).sum().item()
            self.false_positives[class_idx] += (pred_mask & ~target_mask).sum().item()
            self.false_negatives[class_idx] += (~pred_mask & target_mask).sum().item()
    
    def _update_multilabel(self, pred, target):
        """Update metrics for multi-label scenario"""
        pred_binary = (pred >= 0.5).float()
        self.correct += (pred_binary == target).all(dim=1).sum().item()
        self.total += target.size(0)
        
        # Calculate per-class metrics
        for class_idx in range(pred.size(1)):
            pred_mask = pred_binary[:, class_idx] == 1
            target_mask = target[:, class_idx] == 1
            
            self.true_positives[class_idx] += (pred_mask & target_mask).sum().item()
            self.false_positives[class_idx] += (pred_mask & ~target_mask).sum().item()
            self.false_negatives[class_idx] += (~pred_mask & target_mask).sum().item()
    
    def compute(self):
        """
        Compute all metrics
        
        Returns:
            dict: Dictionary containing all computed metrics
        """
        metrics = {}
        
        # Overall accuracy
        metrics['accuracy'] = self.correct / max(self.total, 1) * 100
        
        # Average loss
        if self.num_batches > 0:
            metrics['avg_loss'] = self.accumulated_loss / self.num_batches
        
        # Initialize per-class metric accumulators
        macro_precision = 0
        macro_recall = 0
        macro_f1 = 0
        num_classes = len(self.true_positives)
        
        # Calculate per-class metrics
        for class_idx in range(num_classes):
            tp = self.true_positives[class_idx]
            fp = self.false_positives[class_idx]
            fn = self.false_negatives[class_idx]
            
            # Precision
            precision = tp / max(tp + fp, 1)
            macro_precision += precision
            
            # Recall
            recall = tp / max(tp + fn, 1)
            macro_recall += recall
            
            # F1 Score
            f1 = 2 * precision * recall / max(precision + recall, 1e-6)
            macro_f1 += f1
            
            # Store per-class metrics
            metrics[f'class_{class_idx}_precision'] = precision * 100
            metrics[f'class_{class_idx}_recall'] = recall * 100
            metrics[f'class_{class_idx}_f1'] = f1 * 100
        
        # Calculate macro averages
        metrics['macro_precision'] = (macro_precision / num_classes) * 100
        metrics['macro_recall'] = (macro_recall / num_classes) * 100
        metrics['macro_f1'] = (macro_f1 / num_classes) * 100
        
        return metrics

