#!/usr/bin/env python
import argparse
import os
import torch
from torch.utils.data import DataLoader

from deeplib.config import Config
from deeplib.datasets import build_dataset
from deeplib.models import build_detector
from deeplib.engine import Evaluator

def parse_args():
    parser = argparse.ArgumentParser(description='Test a detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--device', default='cuda:0', help='Device to use')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # Load config
    cfg = Config(args.config)
    
    # Build dataset
    dataset = build_dataset(cfg.data.val)
    data_loader = DataLoader(
        dataset,
        batch_size=cfg.data.samples_per_gpu,
        shuffle=False,
        num_workers=cfg.data.workers_per_gpu,
        collate_fn=dataset.collate_fn,
        pin_memory=True
    )
    
    # Build model
    model = build_detector(cfg.model)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create evaluator
    evaluator = Evaluator(model, data_loader, device=torch.device(args.device))
    
    # Run evaluation
    metrics = evaluator.evaluate()
    
    # Print metrics
    print("Evaluation Results:")
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, list):
            print(f"{metric_name}: {metric_value}")
        else:
            print(f"{metric_name}: {metric_value:.2f}")
    
    # Save results if specified
    if args.out:
        import json
        with open(args.out, 'w') as f:
            json.dump(metrics, f)
        print(f"Results saved to {args.out}")

if __name__ == '__main__':
    main()