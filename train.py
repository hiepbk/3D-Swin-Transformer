import torch
from swin_transformer_3d import SwinTransformer
from dataset import ModelNetDataset
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import Config, Logger, CheckpointManager
from metrics import LabelSmoothingCrossEntropy, FocalLoss, ClassificationMetrics
import time
import random
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed, deterministic=False):
    """Set random seed for reproducibility, similar to MMDetection's approach
    
    Args:
        seed (int): Random seed
        deterministic (bool): Whether to set deterministic options for CUDNN backend
    """
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set environment variables for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if deterministic:
        # Set CUDNN to deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set environment variables for deterministic behavior
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    else:
        # Enable CUDNN benchmark for better performance
        torch.backends.cudnn.benchmark = True

def get_lr_scheduler(optimizer, warmup_iterations, min_lr, total_iterations):
    """Create learning rate scheduler with warmup
    
    Args:
        optimizer: The optimizer to schedule
        warmup_iterations: Number of warmup iterations
        min_lr: Minimum learning rate
        total_iterations: Total number of training iterations
    
    Returns:
        Combined scheduler with warmup and cosine annealing
    """
    # Create warmup scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_iterations
    )
    
    # Create main scheduler (cosine annealing)
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(total_iterations - warmup_iterations),
        eta_min=min_lr
    )
    
    # Combine schedulers
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_iterations]
    )

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
        
    def __call__(self, value):
        if self.best_value is None:
            self.best_value = value
        elif self.mode == 'max':
            if value > self.best_value + self.min_delta:
                self.best_value = value
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'min'
            if value < self.best_value - self.min_delta:
                self.best_value = value
                self.counter = 0
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            self.should_stop = True
            
        return self.should_stop

def main():
    # Load configuration
    cfg = Config.from_file("config.py")
    config = cfg.load_config()

    # Set random seed for reproducibility
    set_seed(config.optimizer_cfg.seed, config.optimizer_cfg.deterministic)
    
    # Initialize logger and checkpoint manager
    logger = Logger(config.log_cfg.log_dir)
    checkpoint_manager = CheckpointManager(config.log_cfg.ckpt_dir)
    
    # Log configuration
    logger.log_config(config._config)
    logger.info(f"Training on device: {device}")
    logger.info(f"Random seed set to: {config.optimizer_cfg.seed}")
    logger.info(f"Deterministic mode: {config.optimizer_cfg.deterministic}")

    # Create datasets and dataloaders
    train_dataset = ModelNetDataset(
        root_dir=config.dataset_cfg.root_dir,
        num_classes=config.dataset_cfg.num_classes,
        num_feat=config.dataset_cfg.num_feat,
        grid_size=config.dataset_cfg.grid_size,
        pc_range=config.dataset_cfg.pc_range,
        split=config.dataset_cfg.train_cfg.split,
    )
    val_dataset = ModelNetDataset(
        root_dir=config.dataset_cfg.root_dir,
        num_classes=config.dataset_cfg.num_classes,
        num_feat=config.dataset_cfg.num_feat,
        grid_size=config.dataset_cfg.grid_size,
        pc_range=config.dataset_cfg.pc_range,
        split=config.dataset_cfg.val_cfg.split,
    )
    
    # Calculate class weights for balanced training
    train_labels = train_dataset.train_labels
    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    logger.info("\nClass distribution and weights:")
    for i, (count, weight) in enumerate(zip(class_counts, class_weights)):
        logger.info(f"Class {i} ({train_dataset.classes[i]}): {count} samples, weight: {weight:.4f}")
    # Log dataset sizes
    logger.info(f"Training set size: {len(train_dataset)} samples")
    logger.info(f"Validation set size: {len(val_dataset)} samples")
    # Set worker seeds for DataLoader
    def worker_init_fn(worker_id):
        worker_seed = config.optimizer_cfg.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.dataset_cfg.train_cfg.batch_size,
        shuffle=config.dataset_cfg.train_cfg.shuffle,
        num_workers=config.dataset_cfg.train_cfg.num_workers,
        pin_memory=config.dataset_cfg.train_cfg.pin_memory,
        drop_last=config.dataset_cfg.train_cfg.drop_last,
        worker_init_fn=worker_init_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.dataset_cfg.val_cfg.batch_size,
        shuffle=config.dataset_cfg.val_cfg.shuffle,
        num_workers=config.dataset_cfg.val_cfg.num_workers,
        pin_memory=config.dataset_cfg.val_cfg.pin_memory,
        drop_last=config.dataset_cfg.val_cfg.drop_last,
        worker_init_fn=worker_init_fn
    )
    
    # Initialize model
    model = SwinTransformer(**config.model_cfg._config).to(device)
    
    # Initialize loss functions with class weights
    criterion = LabelSmoothingCrossEntropy(
        smoothing=config.loss_cfg.ce_smoothing,
        weight=class_weights
    ).to(device)
    
    focal_loss = FocalLoss(
        alpha=config.loss_cfg.focal_alpha,
        gamma=config.loss_cfg.focal_gamma,
        weight=class_weights
    ).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.optimizer_cfg.lr,
        weight_decay=config.optimizer_cfg.weight_decay
    )
    
    scheduler = get_lr_scheduler(
        optimizer,
        config.optimizer_cfg.warmup_iterations,
        config.optimizer_cfg.min_lr,
        len(train_loader) * config.optimizer_cfg.num_epochs
    )
    
    # Training loop
    total_epochs = config.optimizer_cfg.num_epochs
    total_iterations = len(train_loader) * total_epochs
    total_iter = 0
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(total_epochs):
        # Train one epoch
        train_metrics, total_iter = train_one_epoch(
            model,
            train_loader,
            criterion,
            focal_loss,
            optimizer,
            scheduler,
            device,
            epoch,
            total_epochs,
            total_iter,
            total_iterations,
            logger,
            config
        )
        
        # Validate
        val_metrics = val_one_epoch(
            model,
            val_loader,
            criterion,
            focal_loss,
            device,
            epoch,
            total_epochs,
            logger,
            ClassificationMetrics(config.model_cfg.num_classes)
        )
        
        # Save best model
        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            patience_counter = 0
            checkpoint_manager.save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                val_metrics,
                is_best=True
            )
            logger.info(f"Saved best model with validation accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config.optimizer_cfg.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save regular checkpoint
        if (epoch + 1) % config.log_cfg.save_freq == 0:
            checkpoint_manager.save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                val_metrics,
                is_best=False
            )
    
    # Save final model
    checkpoint_manager.save_checkpoint(
        model,
        optimizer,
        scheduler,
        total_epochs-1,
        val_metrics,
        is_best=False
    )
    logger.info("Training completed!")
    
    # Close logger
    logger.close()

def train_one_epoch(model, train_loader, criterion, focal_loss, optimizer, scheduler, device, epoch, total_epochs, total_iter, total_iterations, logger, config):
    model.train()
    metrics = ClassificationMetrics(config.model_cfg.num_classes)
    
    for batch_idx, (points, labels) in enumerate(train_loader):
        points, labels = points.to(device), labels.to(device)
        # print(points.shape)
        
        optimizer.zero_grad()
        outputs = model(points)
        
        # Calculate losses
        ce_loss = criterion(outputs, labels)
        foc_loss = focal_loss(outputs, labels)
        total_loss = ce_loss + config.loss_cfg.focal_weight * foc_loss
        
        total_loss.backward()
        optimizer.step()
        
        # Update learning rate
        scheduler.step()
        
        # Get current learning rate after step
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update metrics
        metrics.update(outputs, labels, total_loss.item())
        
        # Log metrics
        if total_iter % config.log_cfg.log_interval == 0:
            batch_metrics = metrics.compute()
            logger.log_metrics({
                'loss/ce': ce_loss.item(),
                'loss/focal': foc_loss.item(),
                'loss/total': total_loss.item(),
                'metrics/acc': batch_metrics['acc'],
                'metrics/f1': batch_metrics['f1'],
                'lr': current_lr
            }, total_iter, epoch, total_epochs, total_iterations)
            
            # Log progress with detailed iteration information
            logger.info(
                f"Epoch [{epoch}/{total_epochs}], "
                f"Local Iter [{batch_idx}/{len(train_loader)}], "
                f"Global Iter [{total_iter}/{total_iterations}], "
                f"CE Loss: {ce_loss.item():.4f}, Focal Loss: {foc_loss.item():.4f}, "
                f"Total Loss: {total_loss.item():.4f}, Acc: {batch_metrics['acc']:.4f}, "
                f"F1: {batch_metrics['f1']:.4f}, LR: {current_lr:.6f}"
            )
        
        total_iter += 1
    
    # Compute final metrics
    final_metrics = metrics.compute()
    
    # Log epoch completion with detailed metrics
    logger.info(f"\nEpoch [{epoch}/{total_epochs}] completed:")
    logger.info(f"Overall Metrics:")
    logger.info(f"Loss: {final_metrics['loss']:.4f}")
    logger.info(f"Accuracy: {final_metrics['acc']:.4f}")
    logger.info(f"F1 Score: {final_metrics['f1']:.4f}")
    
    logger.info("\nPer-class Metrics:")
    class_names = train_loader.dataset.classes
    for i in range(len(class_names)):
        logger.info(f"\nClass {i} ({class_names[i]}):")
        logger.info(f"  Precision: {final_metrics[f'class_{i}_precision']:.4f}")
        logger.info(f"  Recall: {final_metrics[f'class_{i}_recall']:.4f}")
        logger.info(f"  F1 Score: {final_metrics[f'class_{i}_f1']:.4f}")
        logger.info(f"  Accuracy: {final_metrics[f'class_{i}_acc']:.4f}")
        logger.info(f"  Support: {int(final_metrics[f'class_{i}_support'])}")
    
    return final_metrics, total_iter

def val_one_epoch(model, val_loader, criterion, focal_loss, device, epoch, total_epochs, logger, metrics):
    model.eval()
    metrics.reset()
    
    with torch.no_grad():
        for batch_idx, (points, labels) in enumerate(val_loader):
            points, labels = points.to(device), labels.to(device)
            outputs = model(points)
            
            # Calculate losses
            ce_loss = criterion(outputs, labels)
            foc_loss = focal_loss(outputs, labels)
            total_loss = ce_loss + 0.1 * foc_loss  # Reduced focal loss weight
            
            # Update metrics
            metrics.update(outputs, labels, total_loss.item())
            
            # Log progress
            if batch_idx % 100 == 0:
                batch_metrics = metrics.compute()
                logger.info(
                    f"Validation - Epoch [{epoch}/{total_epochs}], Batch [{batch_idx}/{len(val_loader)}], "
                    f"CE Loss: {ce_loss.item():.4f}, Focal Loss: {foc_loss.item():.4f}, "
                    f"Total Loss: {total_loss.item():.4f}, Acc: {batch_metrics['acc']:.4f}, "
                    f"F1: {batch_metrics['f1']:.4f}"
                )
    
    # Compute final metrics
    final_metrics = metrics.compute()
    
    # Log metrics
    logger.log_metrics({
        'val_loss/ce': final_metrics['loss'],
        'val_loss/focal': final_metrics['loss'],
        'val_loss/total': final_metrics['loss'],
        'val_metrics/acc': final_metrics['acc'],
        'val_metrics/f1': final_metrics['f1']
    }, epoch, epoch, total_epochs, len(val_loader))
    
    # Log detailed metrics
    logger.info(f"\nValidation - Epoch [{epoch}/{total_epochs}] completed:")
    logger.info(f"Overall Metrics:")
    logger.info(f"Loss: {final_metrics['loss']:.4f}")
    logger.info(f"Accuracy: {final_metrics['acc']:.4f}")
    logger.info(f"F1 Score: {final_metrics['f1']:.4f}")
    
    logger.info("\nPer-class Metrics:")
    class_names = val_loader.dataset.classes
    for i in range(len(class_names)):
        logger.info(f"\nClass {i} ({class_names[i]}):")
        logger.info(f"  Precision: {final_metrics[f'class_{i}_precision']:.4f}")
        logger.info(f"  Recall: {final_metrics[f'class_{i}_recall']:.4f}")
        logger.info(f"  F1 Score: {final_metrics[f'class_{i}_f1']:.4f}")
        logger.info(f"  Accuracy: {final_metrics[f'class_{i}_acc']:.4f}")
        logger.info(f"  Support: {int(final_metrics[f'class_{i}_support'])}")
    
    return final_metrics

if __name__ == "__main__":
    main()