import torch
import numpy as np
from tqdm import tqdm
import logging

class Evaluator:
    """
    Evaluator class for model evaluation and inference.
    
    Args:
        model (nn.Module): The model to evaluate
        data_loader (DataLoader): Dataloader for evaluation
        device (torch.device, optional): Device to run evaluation on
    """
    def __init__(self, model, data_loader, device=None):
        self.model = model
        self.data_loader = data_loader
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Setup logging
        self.logger = logging.getLogger('evaluator')
    
    def evaluate(self):
        """
        Evaluate the model on the dataset.
        
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        self.logger.info("Starting evaluation")
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data in tqdm(self.data_loader, desc="Evaluating"):
                # Move data to device
                imgs = data['img'].to(self.device)
                labels = data['gt_label'].to(self.device)
                
                # Get predictions
                preds = self.model(imgs, return_loss=False)
                
                # Store predictions and labels
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        
        # Concatenate all predictions and labels
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_preds, all_labels)
        
        return metrics
    
    def _calculate_metrics(self, preds, labels):
        """
        Calculate evaluation metrics.
        
        Args:
            preds (torch.Tensor): Model predictions
            labels (torch.Tensor): Ground truth labels
            
        Returns:
            dict: Dictionary of metrics
        """
        metrics = {}
        
        # For single-label classification
        if not labels.dim() > 1:
            pred_classes = preds.argmax(dim=1)
            accuracy = (pred_classes == labels).float().mean().item() * 100
            metrics['accuracy'] = accuracy
            
            # Calculate per-class accuracy
            num_classes = preds.size(1)
            per_class_acc = []
            
            for i in range(num_classes):
                class_mask = (labels == i)
                if class_mask.sum() > 0:
                    class_acc = (pred_classes[class_mask] == i).float().mean().item() * 100
                    per_class_acc.append(class_acc)
                else:
                    per_class_acc.append(0)
            
            metrics['per_class_accuracy'] = per_class_acc
            metrics['mean_per_class_accuracy'] = np.mean(per_class_acc)
            
        # For multi-label classification
        else:
            # Convert predictions to binary predictions using threshold of 0.5
            binary_preds = (preds > 0.5).float()
            
            # Calculate accuracy, precision, recall, F1
            correct = (binary_preds == labels).float().mean().item() * 100
            metrics['accuracy'] = correct
            
            # Calculate per-class metrics
            num_classes = preds.size(1)
            precisions = []
            recalls = []
            f1_scores = []
            
            for i in range(num_classes):
                true_positives = ((binary_preds[:, i] == 1) & (labels[:, i] == 1)).sum().item()
                false_positives = ((binary_preds[:, i] == 1) & (labels[:, i] == 0)).sum().item()
                false_negatives = ((binary_preds[:, i] == 0) & (labels[:, i] == 1)).sum().item()
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
            
            metrics['mean_precision'] = np.mean(precisions)
            metrics['mean_recall'] = np.mean(recalls)
            metrics['mean_f1'] = np.mean(f1_scores)
        
        return metrics
    
    def inference(self, imgs):
        """
        Run inference on a batch of images.
        
        Args:
            imgs (torch.Tensor): Batch of images
            
        Returns:
            torch.Tensor: Model predictions
        """
        self.model.eval()
        
        with torch.no_grad():
            imgs = imgs.to(self.device)
            preds = self.model(imgs, return_loss=False)
        
        return preds 