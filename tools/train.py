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
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--no-validate', action='store_true', help='whether not to evaluate the checkpoint during training')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--extra-tag', type=str, default='', help='extra tag for the run')
    args, rest = parser.parse_known_args()
    args.cfg_options = rest
    
    return args

def main():
    args = parse_args()
    
    # Load config
    cfg = Config(args.config, merge_from_args=args.cfg_options)
    
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
            
    trainer = Trainer(cfg)
    trainer.train()

if __name__ == '__main__':
    main()