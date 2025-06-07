import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy loss with class weights"""
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        
    def forward(self, pred, target):
        log_prob = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        if self.weight is not None:
            # Apply class weights to the loss for each sample
            weight = self.weight[target]  # Shape: [batch_size]
            # Expand weight to match the dimensions of the loss
            weight = weight.view(-1, 1)  # Shape: [batch_size, 1]
            # Calculate weighted loss
            loss = -(weight * true_dist * log_prob).sum(dim=-1)
            return loss.mean()
        else:
            return -(true_dist * log_prob).sum(dim=-1).mean()

class FocalLoss(nn.Module):
    """Focal loss with class weights"""
    def __init__(self, alpha=0.25, gamma=2.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class ClassificationMetrics:
    """Class for computing and tracking classification metrics"""
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        """Reset all metrics"""
        self.total_loss = 0.0
        self.total_samples = 0
        self.all_preds = []
        self.all_labels = []
        
    def update(self, 
               outputs: torch.Tensor, 
               labels: torch.Tensor, 
               loss: float):
        """Update metrics with new batch
        
        Args:
            outputs (torch.Tensor): Model outputs
            labels (torch.Tensor): Ground truth labels
            loss (float): Loss value
        """
        self.total_loss += loss
        self.total_samples += labels.size(0)
        
        # Store predictions and labels
        _, preds = torch.max(outputs, 1)
        self.all_preds.extend(preds.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())
        
    def compute(self) -> Dict[str, float]:
        """Compute all metrics
        
        Returns:
            Dict[str, float]: Dictionary of computed metrics
        """
        metrics = {}
        
        # Calculate average loss
        metrics['loss'] = self.total_loss / self.total_samples
        
        # Convert to numpy arrays
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        
        # Get unique classes in the data
        unique_classes = np.unique(np.concatenate([preds, labels]))
        
        # Calculate accuracy
        metrics['acc'] = (preds == labels).mean()
        
        # Calculate F1 score with zero_division=0
        metrics['f1'] = f1_score(labels, preds, average='weighted', zero_division=0)
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, 
            labels=unique_classes,  # Only compute metrics for classes present in the data
            average=None, 
            zero_division=0
        )
        
        # Store per-class metrics
        for i, class_idx in enumerate(unique_classes):
            metrics[f'class_{class_idx}_precision'] = precision[i]
            metrics[f'class_{class_idx}_recall'] = recall[i]
            metrics[f'class_{class_idx}_f1'] = f1[i]
            metrics[f'class_{class_idx}_support'] = support[i]
            
            # Calculate per-class accuracy safely
            class_preds = (preds == class_idx)
            class_labels = (labels == class_idx)
            if class_labels.sum() > 0:  # Only calculate if there are samples for this class
                metrics[f'class_{class_idx}_acc'] = (class_preds & class_labels).sum() / class_labels.sum()
            else:
                metrics[f'class_{class_idx}_acc'] = 0.0
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix
        
        Returns:
            np.ndarray: Confusion matrix
        """
        return confusion_matrix(self.all_labels, self.all_preds)
    
    def get_classification_report(self) -> str:
        """Get detailed classification report
        
        Returns:
            str: Formatted classification report
        """
        metrics = self.compute()
        report = []
        
        # Overall metrics
        report.append(f"Overall Metrics:")
        report.append(f"Loss: {metrics['loss']:.4f}")
        report.append(f"Accuracy: {metrics['acc']:.4f}")
        report.append(f"F1 Score: {metrics['f1']:.4f}")
        report.append("\nPer-class Metrics:")
        
        # Get unique classes from metrics keys
        class_indices = sorted(set(int(k.split('_')[1]) for k in metrics.keys() if k.startswith('class_') and k.endswith('_precision')))
        
        # Per-class metrics
        for class_idx in class_indices:
            report.append(f"\nClass {class_idx}:")
            report.append(f"  Precision: {metrics[f'class_{class_idx}_precision']:.4f}")
            report.append(f"  Recall: {metrics[f'class_{class_idx}_recall']:.4f}")
            report.append(f"  F1 Score: {metrics[f'class_{class_idx}_f1']:.4f}")
            report.append(f"  Accuracy: {metrics[f'class_{class_idx}_acc']:.4f}")
            report.append(f"  Support: {metrics[f'class_{class_idx}_support']}")
        
        return "\n".join(report)

def compute_metrics(outputs: torch.Tensor, 
                   labels: torch.Tensor, 
                   loss: float,
                   num_classes: int) -> Tuple[Dict[str, float], np.ndarray]:
    """Compute metrics for a single batch
    
    Args:
        outputs (torch.Tensor): Model outputs
        labels (torch.Tensor): Ground truth labels
        loss (float): Loss value
        num_classes (int): Number of classes
        
    Returns:
        Tuple[Dict[str, float], np.ndarray]: Metrics dictionary and confusion matrix
    """
    metrics = ClassificationMetrics(num_classes)
    metrics.update(outputs, labels, loss)
    return metrics.compute(), metrics.get_confusion_matrix() 