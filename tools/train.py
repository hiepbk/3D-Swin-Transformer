#!/usr/bin/env python
import argparse
import os
import torch
from torch.utils.data import DataLoader

from deeplib.config import Config
from deeplib.datasets import build_dataset
from deeplib.engine import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', default=None, help='the dir to save logs and models')
    parser.add_argument('--resume-from', default=None, help='the checkpoint file to resume from')
    parser.add_argument('--no-validate', action='store_true', help='whether not to evaluate the checkpoint during training')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--extra-tag', type=str, default='', help='extra tag for the run')
    parser.add_argument('--profile', action='store_true', help='enable profiling for performance analysis')
    args, rest = parser.parse_known_args()
    args.cfg_options = rest
    
    return args

def optimize_pytorch_settings():
    """Optimize PyTorch settings for better performance"""
    # Enable optimized attention if available (PyTorch 2.0+)
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
    
    # Optimize CUDA settings
    if torch.cuda.is_available():
        # Enable TensorFloat-32 for faster training on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Optimize CUDNN
        torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
        
        print(f"CUDA optimizations enabled:")
        print(f"  - TF32: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"  - CUDNN benchmark: {torch.backends.cudnn.benchmark}")
        print(f"  - GPU: {torch.cuda.get_device_name()}")
        print(f"  - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

def main():
    args = parse_args()
    
    # Optimize PyTorch settings
    optimize_pytorch_settings()
    
    # Load config
    cfg = Config(args.config, merge_from_args=args)
    
    # Set work directory
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(args.config))[0], args.extra_tag)
    
    # Create work directory
    os.makedirs(cfg.work_dir, exist_ok=True)
    print(f"Work directory: {cfg.work_dir}")
    
    # Set random seeds
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        if args.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
    # Enable profiling if requested
    if args.profile:
        print("Profiling enabled - training will be slower but provide performance insights")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(cfg.work_dir),
            record_shapes=True,
            with_stack=True
        ) as prof:
            trainer = Trainer(cfg)
            trainer.train()
    else:
        trainer = Trainer(cfg)
        trainer.train()

if __name__ == '__main__':
    main()