import importlib.util
import os
import logging
import json
import torch
from typing import Any, Dict, Optional
from datetime import datetime
import time
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class Config:
    """Configuration class for loading and managing model parameters"""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """Initialize Config with optional dictionary of parameters"""
        self._config = config_dict or {}
        
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """Load configuration from a Python file
        
        Args:
            config_path (str): Path to the configuration file
            
        Returns:
            Config: Configuration object
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        # Load the config file as a module
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # Get all variables from the module that end with '_cfg'
        config_dict = {
            key: value for key, value in config_module.__dict__.items()
            if key.endswith('_cfg')
        }
        
        return cls(config_dict)
    
    def load_config(self) -> 'Config':
        """Get the configuration object
        
        Returns:
            Config: Configuration object
        """
        return self
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value
        
        Args:
            key (str): Configuration key
            default (Any, optional): Default value if key not found. Defaults to None.
            
        Returns:
            Any: Configuration value
        """
        return self._config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Get a configuration value using dictionary-style access
        
        Args:
            key (str): Configuration key
            
        Returns:
            Any: Configuration value
        """
        return self._config[key]
    
    def __getattr__(self, key: str) -> Any:
        """Get a configuration value using dot notation
        
        Args:
            key (str): Configuration key
            
        Returns:
            Any: Configuration value
        """
        if key in self._config:
            value = self._config[key]
            if isinstance(value, dict):
                return Config(value)
            return value
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
    
    def __contains__(self, key: str) -> bool:
        """Check if a configuration key exists
        
        Args:
            key (str): Configuration key
            
        Returns:
            bool: True if key exists, False otherwise
        """
        return key in self._config
    

class Logger:
    """Logger class for training logs and metrics"""
    
    def __init__(self, log_dir: str):
        """Initialize logger with tensorboard support
        
        Args:
            log_dir (str): Directory to store logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Create log file
        self.log_file = os.path.join(log_dir, 'training.log')
        with open(self.log_file, 'w') as f:
            f.write(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Set up file handler
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Initialize metrics storage
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
    def info(self, message: str):
        """Log info message to both file and console
        
        Args:
            message (str): Message to log
        """
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] INFO: {message}"
        print(log_message)
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
        
    def warning(self, message: str):
        """Log warning message to both file and console
        
        Args:
            message (str): Message to log
        """
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] WARNING: {message}"
        print(log_message)
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
        
    def error(self, message: str):
        """Log error message to both file and console
        
        Args:
            message (str): Message to log
        """
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] ERROR: {message}"
        print(log_message)
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
        
    def log_metrics(self, metrics: dict, step: int, epoch: int, total_epochs: int, total_iterations: int):
        """Log metrics to tensorboard and file
        
        Args:
            metrics (dict): Dictionary of metric names and values
            step (int): Current step number
            epoch (int): Current epoch number
            total_epochs (int): Total number of epochs
            total_iterations (int): Total number of iterations
        """
        # Log to tensorboard
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)
        
        # Log to file with consistent format
        metrics_str = ', '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        self.info(f'Epoch [{epoch}/{total_epochs}], {step}/{total_iterations} - {metrics_str}')
        
        # Update metrics storage
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
        
        # Save metrics to file
        metrics_file = os.path.join(self.log_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
            
    def log_config(self, config: Dict[str, Any]):
        """Log configuration parameters to tensorboard and file
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        # Log to tensorboard
        for key, value in config.items():
            if isinstance(value, (int, float, str, bool)):
                self.writer.add_text(f"config/{key}", str(value))
        
        # Save config to JSON file
        config_path = os.path.join(self.log_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        self.info(f"Configuration saved to {config_path}")
    
    def log_model_graph(self, model, dummy_input):
        """Log model architecture to TensorBoard without tracing.
        
        Args:
            model: PyTorch model
            dummy_input: Dummy input tensor for visualization
        """
        try:
            # Use a simpler approach to log model structure
            with torch.no_grad():
                # Get model summary
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                # Log model structure as text
                model_str = str(model)
                self.writer.add_text('model/architecture', model_str)
                
                # Log parameter counts
                self.writer.add_text('model/parameters', 
                    f'Total parameters: {total_params:,}\n'
                    f'Trainable parameters: {trainable_params:,}')
                
                # Log a sample forward pass
                with torch.cuda.amp.autocast(enabled=False):
                    output = model(dummy_input)
                    self.writer.add_text('model/sample_output_shape', 
                        f'Input shape: {dummy_input.shape}\n'
                        f'Output shape: {output.shape}')
                
            self.info("Model architecture logged to TensorBoard")
        except Exception as e:
            self.warning(f"Failed to log model architecture: {str(e)}")
    
    def log_histogram(self, name, values, step):
        """Log histogram of values to tensorboard"""
        try:
            # Convert to numpy array and ensure it's float32
            if isinstance(values, torch.Tensor):
                values = values.detach().cpu().numpy()
            
            # Ensure values are in the correct format
            values = np.asarray(values, dtype=np.float32)
            
            # Remove any NaN or inf values
            values = values[~np.isnan(values) & ~np.isinf(values)]
            
            if len(values) > 0:
                # Create histogram manually
                hist, bin_edges = np.histogram(values, bins=30)
                self.writer.add_histogram_raw(
                    name,
                    min=float(bin_edges[0]),
                    max=float(bin_edges[-1]),
                    num=len(values),
                    sum=float(np.sum(values)),
                    sum_squares=float(np.sum(values**2)),
                    bucket_limits=bin_edges[1:].tolist(),
                    bucket_counts=hist.tolist(),
                    global_step=step
                )
        except Exception as e:
            self.warning(f"Failed to log histogram for {name}: {str(e)}")
    
    def log_images(self, name: str, images: torch.Tensor, epoch: int):
        """Log images to tensorboard
        
        Args:
            name (str): Name of the image group
            images (torch.Tensor): Images to log (N, C, H, W)
            epoch (int): Current epoch number
        """
        self.writer.add_images(name, images, epoch)
    
    def log_learning_rate(self, optimizer: torch.optim.Optimizer, epoch: int):
        """Log learning rate to tensorboard
        
        Args:
            optimizer (torch.optim.Optimizer): Optimizer to get learning rate from
            epoch (int): Current epoch number
        """
        for i, param_group in enumerate(optimizer.param_groups):
            self.writer.add_scalar(f'lr/group_{i}', param_group['lr'], epoch)
    
    def close(self):
        """Close the tensorboard writer"""
        self.writer.close()
        self.info("Logger closed")


class CheckpointManager:
    """Manager for saving and loading model checkpoints"""
    
    def __init__(self, ckpt_dir: str):
        """Initialize checkpoint manager
        
        Args:
            ckpt_dir (str): Directory to save checkpoints
        """
        self.ckpt_dir = ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        
    def save_checkpoint(self, 
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                       epoch: int,
                       metrics: Dict[str, float],
                       is_best: bool = False):
        """Save model checkpoint
        
        Args:
            model (torch.nn.Module): Model to save
            optimizer (torch.optim.Optimizer): Optimizer state
            scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler
            epoch (int): Current epoch number
            metrics (Dict[str, float]): Current metrics
            is_best (bool, optional): Whether this is the best model so far. Defaults to False.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
        # Save latest checkpoint
        latest_path = os.path.join(self.ckpt_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # Save epoch checkpoint
        epoch_path = os.path.join(self.ckpt_dir, f'epoch_{epoch}.pth')
        torch.save(checkpoint, epoch_path)
        
        # Save best checkpoint if applicable
        if is_best:
            best_path = os.path.join(self.ckpt_dir, f'best_epoch_{epoch}.pth')
            torch.save(checkpoint, best_path)
            
    def load_checkpoint(self,
                       model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       ckpt_path: Optional[str] = None) -> Dict[str, Any]:
        """Load model checkpoint
        
        Args:
            model (torch.nn.Module): Model to load checkpoint into
            optimizer (Optional[torch.optim.Optimizer]): Optimizer to load state into
            scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Scheduler to load state into
            ckpt_path (Optional[str]): Path to checkpoint file. If None, loads latest checkpoint.
            
        Returns:
            Dict[str, Any]: Checkpoint data
        """
        if ckpt_path is None:
            ckpt_path = os.path.join(self.ckpt_dir, 'latest.pth')
            
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
            
        checkpoint = torch.load(ckpt_path)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return checkpoint
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint
        
        Returns:
            Optional[str]: Path to latest checkpoint if exists, None otherwise
        """
        latest_path = os.path.join(self.ckpt_dir, 'latest.pth')
        return latest_path if os.path.exists(latest_path) else None
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint
        
        Returns:
            Optional[str]: Path to best checkpoint if exists, None otherwise
        """
        best_path = os.path.join(self.ckpt_dir, 'best.pth')
        return best_path if os.path.exists(best_path) else None
        