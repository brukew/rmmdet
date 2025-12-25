#!/usr/bin/env python
"""
Training script with class weighting support.

This wrapper computes inverse-frequency class weights from the annotation file
and properly injects them into the model config before training.

Usage:
    python tools/train_weighted.py CONFIG --ann-file ANN_FILE [options]
    
Example:
    python tools/train_weighted.py configs/stgcn++/stgcnpp_sails_ntu60p/j.py \
        --ann-file data/sails/single/4class_conf04.pkl \
        --work-dir work_dirs/weighted_test \
        --validate --launcher none
"""

import argparse
import os
import pickle
import sys
import warnings
from collections import Counter
from functools import wraps

import torch

# Monkey-patch torch.load for PyTorch 2.6+ compatibility
_original_torch_load = torch.load
@wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


def compute_class_weights(ann_file: str, split: str = 'train', scale: str = 'linear') -> list:
    """
    Compute inverse-frequency class weights from annotation file.
    
    Args:
        ann_file: Path to annotation pickle file
        split: Which split to compute weights from ('train', 'val', 'test')
        scale: Weight scaling method:
            - 'linear': raw inverse frequency (weight = total / (n_classes * count))
            - 'sqrt': square root of weights (milder, better for small datasets)
            - 'log': log of weights (even milder)
            - 'none': uniform weights (equivalent to no weighting)
    
    Returns:
        List of class weights
    """
    import math
    
    with open(ann_file, 'rb') as f:
        data = pickle.load(f)
    
    annotations = data['annotations']
    split_info = data['split']
    split_ids = set(split_info.get(split, []))
    
    frame_dir_to_ann = {ann['frame_dir']: ann for ann in annotations}
    labels = [frame_dir_to_ann[fd]['label'] for fd in split_ids if fd in frame_dir_to_ann]
    
    counts = Counter(labels)
    num_classes = max(counts.keys()) + 1
    total = len(labels)
    
    weights = []
    for i in range(num_classes):
        count = counts.get(i, 1)
        weight = total / (num_classes * count)
        
        # Apply scaling
        if scale == 'sqrt':
            weight = math.sqrt(weight)
        elif scale == 'log':
            weight = 1 + math.log(weight) if weight > 1 else 1.0
        elif scale == 'none':
            weight = 1.0
        # 'linear' keeps weight as-is
        
        weights.append(weight)
    
    return weights


def parse_args():
    parser = argparse.ArgumentParser(description='Train with class weighting')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--ann-file', required=True, help='annotation file for computing weights')
    parser.add_argument('--work-dir', help='output directory')
    parser.add_argument('--validate', action='store_true', help='evaluate during training')
    parser.add_argument('--test-best', action='store_true', help='test best checkpoint after training')
    parser.add_argument('--launcher', default='none', choices=['none', 'pytorch', 'slurm', 'mpi'])
    parser.add_argument('--total-epochs', type=int, help='override total epochs')
    parser.add_argument('--lr', type=float, help='override learning rate')
    parser.add_argument('--num-classes', type=int, help='override number of classes')
    parser.add_argument('--weight-scale', default='linear', choices=['linear', 'sqrt', 'log', 'none'],
                        help='Weight scaling: linear (default), sqrt (recommended for 5+ classes), log (milder), none (uniform)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='deterministic mode')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    return args


def main():
    args = parse_args()
    
    # Import after torch patching
    import mmcv
    from mmcv import Config
    from mmcv.runner import init_dist, set_random_seed
    from pyskl import __version__
    from pyskl.apis import train_model
    from pyskl.datasets import build_dataset
    from pyskl.models import build_model
    from pyskl.utils import collect_env, get_root_logger
    
    # Load config
    cfg = Config.fromfile(args.config)
    
    # Override ann_file paths
    cfg.data.train.dataset.ann_file = args.ann_file
    cfg.data.val.ann_file = args.ann_file
    if hasattr(cfg.data, 'test'):
        cfg.data.test.ann_file = args.ann_file
    
    # Compute class weights
    print(f"Computing class weights from: {args.ann_file}")
    print(f"Weight scaling: {args.weight_scale}")
    class_weights = compute_class_weights(args.ann_file, split='train', scale=args.weight_scale)
    print(f"Class weights: {[round(w, 4) for w in class_weights]}")
    
    # Inject class weights into model config
    if 'loss_cls' not in cfg.model.cls_head:
        cfg.model.cls_head.loss_cls = dict(type='CrossEntropyLoss', loss_weight=1.0)
    cfg.model.cls_head.loss_cls['class_weight'] = class_weights
    
    # Apply other overrides
    if args.work_dir:
        cfg.work_dir = args.work_dir
    if args.total_epochs:
        cfg.total_epochs = args.total_epochs
    if args.lr:
        cfg.optimizer.lr = args.lr
    if args.num_classes:
        cfg.model.cls_head.num_classes = args.num_classes
    
    # Create work_dir
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    
    # Dump config
    cfg.dump(os.path.join(cfg.work_dir, os.path.basename(args.config)))
    
    # Set up logger
    import time
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    
    # Log environment info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60
    logger.info('Environment info:\n' + dash_line + '\n' + env_info + '\n' + dash_line)
    logger.info(f'Config:\n{cfg.pretty_text}')
    
    # Set random seed
    seed = args.seed
    if seed is None:
        seed = int(torch.randint(0, 2**31, (1,)).item())
    logger.info(f'Set random seed to {seed}, deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    
    # Build model
    model = build_model(cfg.model)
    model.init_weights()
    
    # Build datasets
    datasets = [build_dataset(cfg.data.train)]
    
    # Handle cfg options
    cfg.validate = args.validate
    cfg.test_best = args.test_best
    
    if cfg.get('omnisource'):
        raise NotImplementedError('OmniSource not supported')
    else:
        cfg.workflow = [('train', 1)]
    
    # Init distributed
    distributed = False
    if args.launcher != 'none':
        init_dist(args.launcher, **cfg.dist_params)
        cfg.gpu_ids = [int(os.environ['LOCAL_RANK'])]
        distributed = True
    else:
        cfg.gpu_ids = [0]
    
    # Train
    test_cfg = dict(test_last=False, test_best=args.test_best)
    train_model(model, datasets, cfg, distributed=distributed, validate=args.validate, test=test_cfg)
    
    print("\nTraining complete with class weighting!")


if __name__ == '__main__':
    main()

