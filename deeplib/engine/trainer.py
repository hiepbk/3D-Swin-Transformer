import os
import time
import torch
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
import logging
from tqdm import tqdm
from deeplib.models import build_architecture
from deeplib.datasets import build_dataset
from torch.utils.data import DataLoader
from deeplib.core import ClassificationEvaluator
from deeplib.utils.registry import HOOK_REGISTRY
from torch.optim.lr_scheduler import SequentialLR, LinearLR
from deeplib.utils.utils import track_progress

class Trainer:
    """
    Trainer class for handling the training and validation process.
    
    Args:
        model (nn.Module): The model to train
        cfg (Config): Configuration object
        data_loaders (dict): Dictionary containing 'train' and optionally 'val' dataloaders
        resume_from (str, optional): Path to checkpoint to resume from
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize mixed precision scaler for faster training
        self.use_amp = getattr(cfg, 'use_amp', True)  # Enable by default if CUDA available
        if self.use_amp and torch.cuda.is_available():
            self.scaler = GradScaler()
            print("Mixed precision training enabled")
        else:
            self.scaler = None
            print("Mixed precision training disabled")
            
        # Initialize hooks
        self.hooks = self._build_hooks()

        # Build model
        self.model = self._build_model()
        self.model.to(self.device)
        
        # Build dataloaders first (needed for scheduler calculation)
        self.data_loaders = self._build_dataloader()
        
        # Setup optimizer
        self.optimizer = self._build_optimizer()
        
        # Setup learning rate scheduler (needs dataloader length)
        self.lr_scheduler = self._build_lr_scheduler()
        
        # Initialize training state
        self.epoch = 0
        self.iter = 0
        self.best_val_acc = 0
        self.evaluator = self._build_evaluator()
        

        print("resume from", self.cfg.resume_from)
        # Resume from checkpoint if specified
        if self.cfg.resume_from is not None:
            self._resume_from_checkpoint(self.cfg.resume_from)
            
    def _build_dataloader(self):
        """Build dataloader from config."""
        train_set, val_set = build_dataset(self.cfg)
        
        # Use the dataset's collate_fn if available
        train_collate_fn = getattr(train_set, 'collate_fn', None)
        val_collate_fn = getattr(val_set, 'collate_fn', None)
        
        # Get optimization parameters from config
        train_cfg = self.cfg.dataset.train
        val_cfg = self.cfg.dataset.val
        
        train_loader = DataLoader(
            train_set, 
            batch_size=train_cfg.batch_size, 
            shuffle=train_cfg.shuffle, 
            num_workers=train_cfg.num_workers, 
            pin_memory=train_cfg.pin_memory,
            drop_last=train_cfg.drop_last,
            collate_fn=train_collate_fn,
            persistent_workers=getattr(train_cfg, 'persistent_workers', False),
            prefetch_factor=getattr(train_cfg, 'prefetch_factor', 2)
        )
        
        val_loader = DataLoader(
            val_set, 
            batch_size=val_cfg.batch_size, 
            shuffle=val_cfg.shuffle, 
            num_workers=val_cfg.num_workers, 
            pin_memory=val_cfg.pin_memory,
            drop_last=val_cfg.drop_last,
            collate_fn=val_collate_fn,
            persistent_workers=getattr(val_cfg, 'persistent_workers', False),
            prefetch_factor=getattr(val_cfg, 'prefetch_factor', 2)
        )
        
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
        
        # Compile model for faster training (PyTorch 2.0+)
        if getattr(self.cfg, 'compile_model', False):
            try:
                if hasattr(torch, 'compile'):
                    model = torch.compile(model, mode='max-autotune')
                    print("Model compiled with torch.compile for faster training")
                else:
                    print("torch.compile not available, skipping model compilation")
            except Exception as e:
                print(f"Model compilation failed: {e}, continuing without compilation")
        
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
            # Calculate total iterations for cosine decay after warmup
            total_train_iters = self.cfg.optimizer.num_epochs * len(self.data_loaders['train'])
            if hasattr(lr_config, 'warmup') and lr_config.warmup:
                # T_max = remaining iterations after warmup
                T_max = total_train_iters - lr_config.warmup_iters
            else:
                # No warmup, use all training iterations
                T_max = total_train_iters
            main_scheduler = scheduler_class(self.optimizer, T_max=T_max, eta_min=lr_config.min_lr)
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
        
        # Log available keys for debugging
        self.logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("Model state loaded successfully")
        else:
            self.logger.warning("No model_state_dict found in checkpoint")
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info("Optimizer state loaded successfully")
        else:
            self.logger.warning("No optimizer_state_dict found in checkpoint")
        
        # Load scheduler state if exists
        if 'scheduler_state_dict' in checkpoint and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.logger.info("LR scheduler state loaded successfully")
        
        # Load training state with defaults for missing keys
        self.epoch = checkpoint.get('epoch', 0)
        self.iter = checkpoint.get('iter', 0)
        self.best_val_acc = checkpoint.get('best_val_acc', 0)
        
        self.logger.info(f"Resumed from epoch {self.epoch}, iteration {self.iter}, best_val_acc {self.best_val_acc}")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        self.evaluator.reset()

        self.local_iter = 0
        
        train_loader = self.data_loaders['train']
        
        for batch_data in train_loader:
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            if self.use_amp and self.scaler is not None:
                # Mixed precision forward pass
                with autocast():
                    losses = self.model(batch_data, istrain=True)
                
                # Scale loss and backward pass
                self.scaler.scale(losses.get('Loss')).backward()
                
                # Gradient clipping with scaler
                if hasattr(self.cfg, 'grad_clip'):
                    self.scaler.unscale_(self.optimizer)
                    grad_clip_kwargs = self.cfg.grad_clip.to_dict()
                    clip_grad_norm_(self.model.parameters(), **grad_clip_kwargs)
                
                # Optimizer step with scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision training
                losses = self.model(batch_data, istrain=True)
                losses.get('Loss').backward()
                
                if hasattr(self.cfg, 'grad_clip'):
                    grad_clip_kwargs = self.cfg.grad_clip.to_dict()
                    clip_grad_norm_(self.model.parameters(), **grad_clip_kwargs)
                
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
        from deeplib.utils.utils import track_progress

        if 'val' not in self.data_loaders:
            self.logger.info("No validation dataloader provided. Skipping validation.")
            return 0
        
        self.model.eval()
        self.evaluator.reset()
        data_loader = self.data_loaders['val']
        
        with torch.no_grad():
            for i, batch_data in track_progress(data_loader, "Evaluating", bar_length=25):
                pred, _ = self.model(batch_data, istrain=False)
                self.evaluator.update(pred, batch_data['gt_label'])

        # Get all metrics
        metrics = self.evaluator.evaluate()
        
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