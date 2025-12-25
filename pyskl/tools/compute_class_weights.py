#!/usr/bin/env python
"""
Compute inverse-frequency class weights from a pyskl annotation pickle file.

Usage:
    python tools/compute_class_weights.py data/sails/single/4class_conf04.pkl --split train
    python tools/compute_class_weights.py data/sails/cv/4class_conf04_fold0.pkl --split train

Output format (for bash):
    [0.4479,1.0194,2.0746,3.2847]
"""

import argparse
import pickle
from collections import Counter


def compute_class_weights(ann_file: str, split: str = 'train') -> list:
    """
    Compute inverse-frequency class weights from annotation file.
    
    Formula (same as V-JEPA): weight[i] = total / (num_classes * class_count[i])
    
    Args:
        ann_file: Path to pyskl annotation pickle file
        split: Which split to compute weights from ('train', 'val', 'test')
    
    Returns:
        List of class weights in class order [0, 1, 2, ...]
    """
    with open(ann_file, 'rb') as f:
        data = pickle.load(f)
    
    annotations = data['annotations']
    split_info = data['split']
    
    # Get frame_dirs for the specified split
    split_ids = set(split_info.get(split, []))
    
    if not split_ids:
        raise ValueError(f"Split '{split}' not found or empty in {ann_file}")
    
    # Build lookup and get labels for split
    frame_dir_to_ann = {ann['frame_dir']: ann for ann in annotations}
    labels = [frame_dir_to_ann[fd]['label'] for fd in split_ids if fd in frame_dir_to_ann]
    
    if not labels:
        raise ValueError(f"No samples found for split '{split}' in {ann_file}")
    
    # Count classes
    counts = Counter(labels)
    num_classes = max(counts.keys()) + 1  # Assumes 0-indexed classes
    total = len(labels)
    
    # Compute inverse-frequency weights
    weights = []
    for i in range(num_classes):
        count = counts.get(i, 1)  # Avoid divide by zero
        weight = total / (num_classes * count)
        weights.append(weight)
    
    return weights


def format_weights_for_bash(weights: list, precision: int = 4) -> str:
    """Format weights as a bash-compatible array string."""
    formatted = [f"{w:.{precision}f}" for w in weights]
    return "[" + ",".join(formatted) + "]"


def main():
    parser = argparse.ArgumentParser(
        description="Compute inverse-frequency class weights from annotation pickle"
    )
    parser.add_argument('ann_file', type=str, help='Path to annotation pickle file')
    parser.add_argument('--split', type=str, default='train',
                        help='Split to compute weights from (default: train)')
    parser.add_argument('--precision', type=int, default=4,
                        help='Decimal precision for weights (default: 4)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed class distribution')
    
    args = parser.parse_args()
    
    # Load and analyze
    with open(args.ann_file, 'rb') as f:
        data = pickle.load(f)
    
    annotations = data['annotations']
    split_info = data['split']
    split_ids = set(split_info.get(args.split, []))
    
    frame_dir_to_ann = {ann['frame_dir']: ann for ann in annotations}
    labels = [frame_dir_to_ann[fd]['label'] for fd in split_ids if fd in frame_dir_to_ann]
    
    counts = Counter(labels)
    num_classes = max(counts.keys()) + 1
    total = len(labels)
    
    weights = compute_class_weights(args.ann_file, args.split)
    
    if args.verbose:
        import sys
        print(f"File: {args.ann_file}", file=sys.stderr)
        print(f"Split: {args.split} ({total} samples)", file=sys.stderr)
        print(f"Class distribution:", file=sys.stderr)
        for i in range(num_classes):
            c = counts.get(i, 0)
            pct = c / total * 100 if total > 0 else 0
            print(f"  Class {i}: {c:4d} ({pct:5.1f}%) -> weight {weights[i]:.4f}", file=sys.stderr)
        print(file=sys.stderr)
    
    # Output just the formatted weights (for bash capture)
    print(format_weights_for_bash(weights, args.precision))


if __name__ == '__main__':
    main()

