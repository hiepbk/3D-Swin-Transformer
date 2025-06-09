import os
import time
import torch
import logging
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from deeplib.utils.registry import HOOK_REGISTRY

class Hook:
    def __init__(self):
        pass
        
    def before_run(self, trainer):
        pass
        
    def after_run(self, trainer):
        pass
        
    def before_epoch(self, trainer):
        pass
        
    def after_epoch(self, trainer, metrics, mode='train'):
        pass
        
    def before_iter(self, trainer):
        pass
        
    def after_iter(self, trainer):
        pass

@HOOK_REGISTRY.register_module()
class LoggerHook(Hook):
    def __init__(self, work_dir, save_dir='logs', log_freq=100, val_epoch_interval=1):
        super().__init__()
        self.work_dir = work_dir
        self.save_dir = os.path.join(work_dir, save_dir)
        self.log_freq = log_freq
        self.val_epoch_interval = val_epoch_interval
        os.makedirs(self.save_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.save_dir)
        
        # Configure logger
        self.logger = logging.getLogger('trainer')
        self.logger.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Create file handler
        log_file = os.path.join(self.save_dir, f'train_{time.strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def before_run(self, trainer):
        self.logger.info('Start training')
        # print model network structure
        self.logger.info(f'Model network structure: {trainer.model}')
    
    def after_train_iter(self, trainer, **kwargs):
        batch_metrics = kwargs.get('batch_metrics', {})
        if (trainer.iter + 1) % self.log_freq == 0:
            # Get current learning rate
            current_lr = trainer.optimizer.param_groups[0]['lr']
            
            # Build log message with all loss components
            log_msg = (
                f"Epoch: [{trainer.epoch+1}/{trainer.cfg.optimizer.num_epochs}] "
                f"Iter: [{trainer.local_iter+1}/{len(trainer.data_loaders['train'])}] "
                f"LR: {current_lr:.6f} "
            )
    
            # Add all loss components to log message and tensorboard
            for k, v in batch_metrics.items():
                if isinstance(v, (int, float)):
                    log_msg += f"{k}: {v:.4f} "
                    self.writer.add_scalar(f'train/{k}', v, trainer.iter)
            
            
            self.writer.add_scalar('optimizer/learning_rate', current_lr, trainer.iter)
            self.logger.info(log_msg)

    def after_train_epoch(self, trainer, **kwargs):
        if self.val_epoch_interval is not None and (trainer.epoch + 1) % self.val_epoch_interval == 0:
            trainer.evaluate()
    
    def after_val_epoch(self, trainer, metrics):
        self.logger.info(f'Epoch {trainer.epoch + 1} validation completed')
        
        # Log all metrics
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.logger.info(f'{k}: {v:.4f}')
                self.writer.add_scalar(f'val/{k}', v, trainer.epoch)
                

    
    def after_run(self, trainer):
        self.logger.info('Training completed')
        self.writer.close() 

@HOOK_REGISTRY.register_module()
class CheckpointHook(Hook):
    def __init__(self, work_dir, save_dir='ckpts', save_freq=10):
        super().__init__()
        self.work_dir = work_dir
        self.save_dir = os.path.join(work_dir, save_dir)
        self.save_freq = save_freq
        self.last_epoch = 0
        os.makedirs(self.save_dir, exist_ok=True)

    def after_train_epoch(self, trainer, **kwargs):
        if (trainer.epoch + 1) % self.save_freq == 0:
            if trainer.epoch > self.last_epoch:
                self.last_epoch = trainer.epoch
                # save the last checkpoint
                last_checkpoint = {
                    'epoch': trainer.epoch,
                    'iter': trainer.iter,
                    'model_state_dict': trainer.model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'best_val_acc': trainer.best_val_acc
                }
                save_path = os.path.join(self.save_dir, f'last.pth')
                torch.save(last_checkpoint, save_path)
                trainer.logger.info(f'Last checkpoint saved to {save_path}') 

            checkpoint = {
                'epoch': trainer.epoch,
                'iter': trainer.iter,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'best_val_acc': trainer.best_val_acc
            }
            if trainer.lr_scheduler is not None:
                checkpoint['scheduler_state_dict'] = trainer.lr_scheduler.state_dict()
            
            save_path = os.path.join(self.save_dir, f'epoch_{trainer.epoch}.pth')
            torch.save(checkpoint, save_path)
            trainer.logger.info(f'Checkpoint saved to {save_path}') 

@HOOK_REGISTRY.register_module()
class LRSchedulerHook(Hook):
    def __init__(self):
        super().__init__()
        self.step_on_iter = False
        self.step_on_epoch = False
    
    def before_run(self, trainer):
        # Determine when to step based on scheduler type
        if trainer.lr_scheduler is not None:
            scheduler_name = trainer.lr_scheduler.__class__.__name__
            if scheduler_name in ['SequentialLR', 'OneCycleLR']:
                self.step_on_iter = True
                self.step_on_epoch = False
            else:
                self.step_on_iter = False
                self.step_on_epoch = True
    
    def after_train_iter(self, trainer, **kwargs):
        # Step scheduler after each iteration for specific schedulers
        if self.step_on_iter and trainer.lr_scheduler is not None:
                trainer.lr_scheduler.step()
    
    def after_train_epoch(self, trainer, **kwargs):
        # Step scheduler after each epoch for most schedulers
        if self.step_on_epoch and trainer.lr_scheduler is not None:
                trainer.lr_scheduler.step()

@HOOK_REGISTRY.register_module()
class OptimizerHook(Hook):
    def after_train_iter(self, trainer, **kwargs):
        # Gradient clipping is already handled in the trainer's training loop
        # No additional operations needed here for now
        pass

