import os
import time
import torch
from torch.nn.utils import clip_grad_norm_
import logging
from tqdm import tqdm
from deeplib.models import build_architecture
from deeplib.datasets import build_dataset
from torch.utils.data import DataLoader
from deeplib.core import ClassificationEvaluator
from deeplib.utils.registry import HOOK_REGISTRY
from torch.optim.lr_scheduler import SequentialLR, LinearLR

class Trainer:
    """
    Trainer class for handling the training and validation process.
    
    Args:
        model (nn.Module): The model to train
        cfg (Config): Configuration object
        data_loaders (dict): Dictionary containing 'train' and optionally 'val' dataloaders
        resume_from (str, optional): Path to checkpoint to resume from
    """
    def __init__(self, cfg, resume_from=None):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Initialize hooks
        self.hooks = self._build_hooks()

        # Build model
        self.model = self._build_model()
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._build_optimizer()
        
        # Setup learning rate scheduler
        self.lr_scheduler = self._build_lr_scheduler()
        
        # Build dataloaders
        self.data_loaders = self._build_dataloader()
        
        # Initialize training state
        self.epoch = 0
        self.iter = 0
        self.best_val_acc = 0
        
        self.evaluator = self._build_evaluator()
        
        
        
        # Resume from checkpoint if specified
        if resume_from is not None:
            self._resume_from_checkpoint(resume_from)
            
    def _build_dataloader(self):
        """Build dataloader from config."""
        train_set, val_set = build_dataset(self.cfg)
        
        # Use the dataset's collate_fn if available
        train_collate_fn = getattr(train_set, 'collate_fn', None)
        val_collate_fn = getattr(val_set, 'collate_fn', None)
        
        train_loader = DataLoader(train_set, 
                              batch_size=self.cfg.dataset.train.batch_size, 
                              shuffle=self.cfg.dataset.train.shuffle, 
                              num_workers=self.cfg.dataset.train.num_workers, 
                              pin_memory=self.cfg.dataset.train.pin_memory,
                              collate_fn=train_collate_fn)
        val_loader = DataLoader(val_set, 
                            batch_size=self.cfg.dataset.val.batch_size, 
                            shuffle=self.cfg.dataset.val.shuffle, 
                            num_workers=self.cfg.dataset.val.num_workers, 
                            pin_memory=self.cfg.dataset.val.pin_memory,
                            collate_fn=val_collate_fn)
        data_loaders = {'train': train_loader, 'val': val_loader}

        self.logger.info(f'Train set: {len(train_set)}')
        self.logger.info(f'Val set: {len(val_set)}')
        self.logger.info(f'{len(train_set.classes)} Classes: {train_set.classes}')
        return data_loaders
    
    def _build_evaluator(self):
        """Build evaluator from config."""
        class_names = self.cfg.class_names
        evaluator = ClassificationEvaluator(class_names)
        return evaluator
    
    def _build_model(self):
        """Build model from config."""
        model = build_architecture(self.cfg)
        model.to(self.device)
        return model
    
    def _build_optimizer(self):
        """Build optimizer from config."""
        optimizer_cfg = self.cfg.optimizer
        
        if optimizer_cfg.name == 'SGD':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=optimizer_cfg.lr,
                momentum=optimizer_cfg.momentum,
                weight_decay=optimizer_cfg.weight_decay
            )
        elif optimizer_cfg.name == 'Adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=optimizer_cfg.lr,
                weight_decay=optimizer_cfg.weight_decay
            )
        elif optimizer_cfg.name == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=optimizer_cfg.lr,
                weight_decay=optimizer_cfg.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer name: {optimizer_cfg.name}")
        
        return optimizer
    
    def _build_lr_scheduler(self):
        """Build learning rate scheduler from config."""

        lr_config = self.cfg.lr_config
        
        # Get the main scheduler class
        try:
            scheduler_class = getattr(torch.optim.lr_scheduler, lr_config.policy)
        except AttributeError:
            raise ValueError(f"Unsupported lr policy: {lr_config.policy}")
        
        # Build main scheduler based on policy
        if lr_config.policy == 'MultiStepLR': 
            main_scheduler = scheduler_class(self.optimizer, milestones=lr_config.step, gamma=lr_config.gamma)
        elif lr_config.policy == 'CosineAnnealingLR':
            # Adjust T_max if warmup is used
            total_iters = self.cfg.optimizer.num_epochs
            if hasattr(lr_config, 'warmup') and lr_config.warmup:
                total_iters -= lr_config.warmup_iters
            main_scheduler = scheduler_class(self.optimizer, T_max=total_iters, eta_min=lr_config.min_lr)
        elif lr_config.policy == 'OneCycleLR':
            main_scheduler = scheduler_class(
                self.optimizer, max_lr=lr_config.max_lr,
                total_steps=self.cfg.optimizer.num_epochs * len(self.data_loaders['train']),
                pct_start=lr_config.pct_start, anneal_strategy=lr_config.anneal_strategy)
        elif lr_config.policy == 'LinearLR':
            main_scheduler = scheduler_class(
                self.optimizer, start_factor=lr_config.start_factor,
                end_factor=lr_config.end_factor, total_iters=lr_config.total_iters)
        else:
            main_scheduler = scheduler_class(self.optimizer)
        
        # Add warmup if specified
        if hasattr(lr_config, 'warmup') and lr_config.warmup:
            if lr_config.warmup == 'linear':
                warmup_scheduler = LinearLR(
                    self.optimizer, start_factor=lr_config.warmup_ratio,
                    end_factor=1.0, total_iters=lr_config.warmup_iters)
                return SequentialLR(
                    self.optimizer, schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[lr_config.warmup_iters])
            else:
                raise ValueError(f"Unsupported warmup type: {lr_config.warmup}")
        
        return main_scheduler
    
    def _resume_from_checkpoint(self, checkpoint_path):
        """Resume training from checkpoint."""
        self.logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if exists
        if 'scheduler_state_dict' in checkpoint and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.epoch = checkpoint['epoch']
        self.iter = checkpoint['iter']
        self.best_val_acc = checkpoint.get('best_val_acc', 0)
        
        self.logger.info(f"Resumed from epoch {self.epoch}, iteration {self.iter}")
    
    def _save_checkpoint(self, filename, save_optimizer=True, is_best=False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'iter': self.iter,
            'model_state_dict': self.model.state_dict(),
            'best_val_acc': self.best_val_acc
        }
        
        if save_optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            if self.lr_scheduler is not None:
                checkpoint['scheduler_state_dict'] = self.lr_scheduler.state_dict()
        
        checkpoint_path = os.path.join(self.cfg.work_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        if is_best:
            best_path = os.path.join(self.cfg.work_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved to {best_path}")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        self.evaluator.reset()

        self.local_iter = 0
        
        train_loader = self.data_loaders['train']
        
        for batch_data in train_loader:
            # Forward pass
            self.optimizer.zero_grad()
            # loss is now a dict of loss components from ClassifyLoss
            losses = self.model(batch_data, istrain=True)
            
            # Extract the total loss tensor for backpropagation
            losses.get('Loss').backward()
            
            if hasattr(self.cfg, 'grad_clip'):
                clip_grad_norm_(self.model.parameters(), **self.cfg.grad_clip)
            
            self.optimizer.step()
            
            # Update metrics - convert tensor values to floats for logging
            batch_metrics = {}
            for k, v in losses.items():
                if torch.is_tensor(v):
                    batch_metrics[k] = v.item()
                else:
                    batch_metrics[k] = v
            
            # Call after_train_iter hooks
            for hook in self.hooks:
                if hasattr(hook, 'after_train_iter'):
                    hook.after_train_iter(self, batch_metrics=batch_metrics)
            
            self.iter += 1
            self.local_iter += 1
    
    def evaluate(self):
        """Evaluate the model."""

        if 'val' not in self.data_loaders:
            self.logger.info("No validation dataloader provided. Skipping validation.")
            return 0
        
        self.model.eval()
        self.evaluator.reset()
        data_loader = self.data_loaders['val']
        
        with torch.no_grad():
            for i, batch_data in enumerate(data_loader):
                pred, _ = self.model(batch_data, istrain=False)
                self.evaluator.update(pred, batch_data['gt_label'])

        # Get all metrics
        metrics = self.evaluator.compute()
        
        # Call after_val_epoch hooks
        for hook in self.hooks:
            if hasattr(hook, 'after_val_epoch'):
                hook.after_val_epoch(self, metrics)
        
        return metrics
    
    def call_hook(self, fn_name):
        """Call all hooks with the given function name."""
        for hook in self.hooks:
            if hasattr(hook, fn_name):
                getattr(hook, fn_name)(self)

    def _build_hooks(self):
        """Build hooks from config."""
        hooks = []
        for hook_cfg in self.cfg.hooks:
            hook_type = hook_cfg.pop('type')
            # Only pass work_dir to LoggerHook and CheckpointHook
            if hook_type in ['LoggerHook', 'CheckpointHook']:
                hook_cfg.update(work_dir=self.cfg.work_dir)
            hook = HOOK_REGISTRY.get(hook_type)(**hook_cfg)

            if hook_type == 'LoggerHook':
                self.logger = hook.logger

            hooks.append(hook)

        return hooks

    def train(self):
        """Training process"""
        self.call_hook('before_run')
        
        for epoch in range(self.cfg.optimizer.num_epochs):
            self.epoch = epoch
            self.call_hook('before_epoch')
            self.train_epoch()
            self.call_hook('after_train_epoch')
            
        self.call_hook('after_run') 