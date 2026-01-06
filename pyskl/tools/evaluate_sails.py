#!/usr/bin/env python3
"""
Comprehensive evaluation script for SAILS models.

Supports two task types:
- RMM Classification (4-class): Predicting which RMM type a clip contains
- TAL (5-class): Predicting RMM type or background for temporal action localization

Computes metrics:
- Top-1 and Top-2 accuracy
- Macro and Weighted F1, Precision, Recall
- Cohen's Kappa
- Per-class F1, Precision, Recall
- For TAL: RMM vs Background binary metrics (F1, Precision, Recall, AUC)
- Confusion matrices (saved as PNG)
- Prediction CSVs

Usage:
    # Evaluate RMM classification (4-class)
    python tools/evaluate_sails.py config.py -C checkpoint.pth --split val --output-dir eval_val --task rmm

    # Evaluate TAL (5-class with background)
    python tools/evaluate_sails.py config.py -C checkpoint.pth --split val --output-dir eval_val --task tal
"""

import argparse
import json
import os
import os.path as osp
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# PyTorch 2.6+ security fix
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import mmcv
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

try:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix,
        cohen_kappa_score,
        roc_auc_score,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not installed. Install with: pip install scikit-learn")

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("Warning: matplotlib/seaborn not installed. Confusion matrix plots will be skipped.")


# Default class names for SAILS 4-class RMM classification
DEFAULT_CLASS_NAMES_4 = ["hands flapping", "jumping", "rocking", "spinning"]
# Default class names for SAILS 5-class TAL (includes background)
DEFAULT_CLASS_NAMES_5 = ["hands flapping", "jumping", "rocking", "spinning", "background"]


def parse_args():
    parser = argparse.ArgumentParser(description='Comprehensive SAILS model evaluation')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('-C', '--checkpoint', required=True, help='Checkpoint file path')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                        help='Which split to evaluate on (default: test)')
    parser.add_argument('--output-dir', required=True, help='Directory to save evaluation results')
    parser.add_argument('--task', default='auto', choices=['rmm', 'tal', 'auto'],
                        help='Task type: rmm (4-class), tal (5-class with background), auto (detect from num_classes)')
    parser.add_argument('--launcher', choices=['pytorch', 'none'], default='none',
                        help='Job launcher (default: none for single GPU)')
    parser.add_argument('--num-clips', type=int, default=10,
                        help='Number of clips per video for testing (default: 10)')
    parser.add_argument('--class-names', nargs='+', default=None,
                        help='Class names in order. Default depends on --task')
    parser.add_argument('--device', default='cuda:0', help='Device to use (default: cuda:0)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config options. Format: key=value key2=value2 ...')
    return parser.parse_args()


def top_k_accuracy(scores, labels, topk=(1, 2)):
    """Compute top-k accuracy for multiple k values."""
    results = {}
    maxk = max(topk)
    
    scores = np.array(scores)
    labels = np.array(labels)
    
    # Get top-k predictions
    top_k_preds = np.argsort(scores, axis=1)[:, ::-1][:, :maxk]
    
    for k in topk:
        correct = 0
        for i, label in enumerate(labels):
            if label in top_k_preds[i, :k]:
                correct += 1
        results[f'top{k}_acc'] = correct / len(labels) * 100
    
    return results


def compute_all_metrics(y_true, y_pred, y_scores, class_names, prefix='', task='rmm'):
    """
    Compute comprehensive metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_scores: Prediction scores (N x num_classes)
        class_names: List of class names
        prefix: Prefix for metric keys (e.g., 'clip_')
        task: 'rmm' for 4-class RMM classification, 'tal' for 5-class TAL with background
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    
    # Top-k accuracy
    topk_results = top_k_accuracy(y_scores, y_true, topk=(1, 2))
    metrics[f'{prefix}top1_acc'] = topk_results['top1_acc']
    metrics[f'{prefix}top2_acc'] = topk_results['top2_acc']
    
    # F1, Precision, Recall (all classes)
    metrics[f'{prefix}macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
    metrics[f'{prefix}weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    metrics[f'{prefix}macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
    metrics[f'{prefix}macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100
    
    # Cohen's Kappa
    metrics[f'{prefix}cohens_kappa'] = cohen_kappa_score(y_true, y_pred)
    
    # ========== RMM vs Background Binary Metrics (TAL only) ==========
    if task == 'tal':
        # Assumes class 4 = background, classes 0-3 = RMM
        BACKGROUND_CLASS = 4
        
        # Binary labels: 1 = RMM (any of classes 0-3), 0 = Background (class 4)
        y_true_binary = (y_true != BACKGROUND_CLASS).astype(int)
        y_pred_binary = (y_pred != BACKGROUND_CLASS).astype(int)
        
        # Binary scores: sum of RMM class probabilities
        if y_scores.shape[1] > BACKGROUND_CLASS:
            rmm_scores = y_scores[:, :BACKGROUND_CLASS].sum(axis=1)  # Sum classes 0-3
        else:
            rmm_scores = 1 - y_scores[:, -1]  # Fallback
        
        # Binary classification metrics
        metrics[f'{prefix}rmm_vs_bg_accuracy'] = (y_true_binary == y_pred_binary).mean() * 100
        metrics[f'{prefix}rmm_vs_bg_f1'] = f1_score(y_true_binary, y_pred_binary, pos_label=1, zero_division=0) * 100
        metrics[f'{prefix}rmm_vs_bg_precision'] = precision_score(y_true_binary, y_pred_binary, pos_label=1, zero_division=0) * 100
        metrics[f'{prefix}rmm_vs_bg_recall'] = recall_score(y_true_binary, y_pred_binary, pos_label=1, zero_division=0) * 100
        
        # AUC-ROC for RMM detection
        try:
            # Need both classes present for AUC
            if len(np.unique(y_true_binary)) > 1:
                metrics[f'{prefix}rmm_vs_bg_auc'] = roc_auc_score(y_true_binary, rmm_scores) * 100
            else:
                metrics[f'{prefix}rmm_vs_bg_auc'] = 0.0
        except Exception:
            metrics[f'{prefix}rmm_vs_bg_auc'] = 0.0
        
        # Error rates
        # False alarm: Background predicted as RMM
        bg_mask = y_true_binary == 0
        if bg_mask.sum() > 0:
            metrics[f'{prefix}bg_false_alarm_rate'] = (y_pred_binary[bg_mask] == 1).mean() * 100
        else:
            metrics[f'{prefix}bg_false_alarm_rate'] = 0.0
        
        # Miss rate: RMM predicted as Background
        rmm_mask = y_true_binary == 1
        if rmm_mask.sum() > 0:
            metrics[f'{prefix}rmm_miss_rate'] = (y_pred_binary[rmm_mask] == 0).mean() * 100
        else:
            metrics[f'{prefix}rmm_miss_rate'] = 0.0
        
        # RMM-only Macro F1 (excluding background)
        rmm_indices = list(range(BACKGROUND_CLASS))  # Classes 0-3
        rmm_mask_multiclass = np.isin(y_true, rmm_indices)
        if rmm_mask_multiclass.sum() > 0:
            y_true_rmm = y_true[rmm_mask_multiclass]
            y_pred_rmm = y_pred[rmm_mask_multiclass]
            metrics[f'{prefix}rmm_only_macro_f1'] = f1_score(y_true_rmm, y_pred_rmm, average='macro', zero_division=0) * 100
            metrics[f'{prefix}rmm_only_macro_precision'] = precision_score(y_true_rmm, y_pred_rmm, average='macro', zero_division=0) * 100
            metrics[f'{prefix}rmm_only_macro_recall'] = recall_score(y_true_rmm, y_pred_rmm, average='macro', zero_division=0) * 100
        else:
            metrics[f'{prefix}rmm_only_macro_f1'] = 0.0
            metrics[f'{prefix}rmm_only_macro_precision'] = 0.0
            metrics[f'{prefix}rmm_only_macro_recall'] = 0.0
    
    # ========== Per-class metrics ==========
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    for i, name in enumerate(class_names):
        # For TAL, skip background class in per-class metrics
        if task == 'tal' and 'background' in name.lower():
            continue
        if i < len(per_class_f1):
            metrics[f'{prefix}f1_{name}'] = per_class_f1[i] * 100
            metrics[f'{prefix}prec_{name}'] = per_class_prec[i] * 100
            metrics[f'{prefix}rec_{name}'] = per_class_rec[i] * 100
    
    return metrics


def plot_confusion_matrix(cm, class_names, output_path, title='Confusion Matrix'):
    """Plot and save confusion matrix."""
    if not HAS_PLOT:
        return
    
    plt.figure(figsize=(10, 8))
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)
    
    # Plot with both counts and percentages
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar_kws={'label': 'Proportion'}
    )
    
    # Add raw counts as secondary annotation
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j + 0.5, i + 0.7, f'({cm[i, j]})',
                    ha='center', va='center', fontsize=8, color='gray')
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved confusion matrix: {output_path}")


def save_predictions_csv(predictions, output_path):
    """Save predictions to CSV file."""
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['segment_id', 'true_label', 'pred_label', 'true_class', 'pred_class']
        n_classes = len(predictions[0]['scores'])
        for i in range(n_classes):
            header.append(f'score_class{i}')
        writer.writerow(header)
        
        # Data
        for pred in predictions:
            row = [
                pred['segment_id'],
                pred['true_label'],
                pred['pred_label'],
                pred['true_class'],
                pred['pred_class'],
            ]
            row.extend([f'{s:.4f}' for s in pred['scores']])
            writer.writerow(row)
    
    print(f"  Saved predictions CSV: {output_path}")


def run_inference(model, data_loader, device):
    """Run inference and collect results."""
    model.eval()
    results = []
    
    prog_bar = mmcv.ProgressBar(len(data_loader))
    
    with torch.no_grad():
        for batch in data_loader:
            # Handle both PoseC3D (imgs) and STGCN++ (keypoint) inputs
            if 'imgs' in batch:
                input_data = batch['imgs'].to(device)
            elif 'keypoint' in batch:
                input_data = batch['keypoint'].to(device)
            else:
                raise KeyError(f"Expected 'imgs' or 'keypoint' in batch, got: {batch.keys()}")
            
            label = batch['label'].cpu().numpy()
            
            # Forward pass
            output = model(input_data, return_loss=False)
            
            # Handle both tensor and numpy array outputs
            if isinstance(output, torch.Tensor):
                # output shape: (batch, num_classes) or (batch, num_clips, num_classes)
                if output.ndim == 3:
                    # Average over clips
                    output = output.mean(dim=1)
                scores = output.cpu().numpy()
            else:
                # Already numpy array
                scores = np.array(output)
                if scores.ndim == 3:
                    scores = scores.mean(axis=1)
            
            for i in range(len(label)):
                results.append({
                    'scores': scores[i],
                    'label': label[i],
                })
            
            prog_bar.update()
    
    return results


def main():
    args = parse_args()
    
    if not HAS_SKLEARN:
        print("Error: scikit-learn is required. Install with: pip install scikit-learn")
        sys.exit(1)
    
    # Load config
    cfg = Config.fromfile(args.config)
    
    # Merge cfg_options if provided
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    # Determine task type and class names
    # First check if model has num_classes defined
    num_classes_from_model = cfg.model.get('cls_head', {}).get('num_classes', None)
    
    # Auto-detect task if not specified
    if args.task == 'auto':
        if num_classes_from_model == 5:
            task = 'tal'
        elif num_classes_from_model == 4:
            task = 'rmm'
        else:
            # Default to rmm if can't determine
            task = 'rmm'
        print(f"Auto-detected task: {task} (num_classes={num_classes_from_model})")
    else:
        task = args.task
    
    # Get class names
    if args.class_names:
        class_names = args.class_names
    elif task == 'tal':
        class_names = DEFAULT_CLASS_NAMES_5
    else:
        class_names = DEFAULT_CLASS_NAMES_4
    
    num_classes = len(class_names)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=" * 60)
    print(f"SAILS Model Evaluation")
    print(f"=" * 60)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Task: {task}")
    print(f"Output: {output_dir}")
    print(f"Classes ({num_classes}): {class_names}")
    print()
    
    # Load annotation file to get segment IDs
    # Prefer the ann_file from data.{split} which is set via --cfg-options
    # over the top-level cfg.ann_file which may be a config default
    ann_file = cfg.data[args.split].get('ann_file', cfg.get('ann_file'))
    # Also update data config so dataset builder uses the right file
    cfg.data[args.split]['ann_file'] = ann_file
    print(f"Loading annotations from: {ann_file}")
    
    with open(ann_file, 'rb') as f:
        ann_data = pickle.load(f)
    
    split_ids = set(ann_data['split'].get(args.split, []))
    annotations = [a for a in ann_data['annotations'] if a['frame_dir'] in split_ids]
    
    # Create segment_id mapping
    segment_id_map = {i: a['frame_dir'] for i, a in enumerate(annotations)}
    
    print(f"Found {len(annotations)} samples in '{args.split}' split")
    
    # Build model
    from pyskl.models import build_model
    model = build_model(cfg.model)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    # Move to device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Build dataloader
    from pyskl.datasets import build_dataset, build_dataloader
    
    # Use test pipeline
    dataset_cfg = cfg.data[args.split].copy()
    dataset = build_dataset(dataset_cfg)
    
    data_loader = build_dataloader(
        dataset,
        videos_per_gpu=1,
        workers_per_gpu=4,
        shuffle=False,
    )
    
    # Run inference
    print(f"\nRunning inference on {len(dataset)} samples...")
    results = run_inference(model, data_loader, device)
    
    # Process results
    print(f"\nProcessing results...")
    
    # Clip-level results
    clip_scores = np.array([r['scores'] for r in results])
    clip_labels = np.array([r['label'] for r in results])
    clip_preds = np.argmax(clip_scores, axis=1)
    
    # Build predictions list with segment IDs
    predictions_clip = []
    for i, r in enumerate(results):
        segment_id = segment_id_map.get(i, f"sample_{i}")
        predictions_clip.append({
            'segment_id': segment_id,
            'true_label': int(r['label']),
            'pred_label': int(clip_preds[i]),
            'true_class': class_names[r['label']] if r['label'] < len(class_names) else f"class_{r['label']}",
            'pred_class': class_names[clip_preds[i]] if clip_preds[i] < len(class_names) else f"class_{clip_preds[i]}",
            'scores': r['scores'].tolist(),
        })
    
    # Compute clip-level metrics
    print("\n--- Clip-Level Metrics ---")
    clip_metrics = compute_all_metrics(clip_labels, clip_preds, clip_scores, class_names, prefix='clip_', task=task)
    
    for key, value in clip_metrics.items():
        if 'kappa' in key:
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value:.2f}%")
    
    # Combine all metrics (clip-level only, no video-level)
    all_metrics = clip_metrics.copy()
    all_metrics['num_clips'] = len(clip_labels)
    all_metrics['split'] = args.split
    all_metrics['task'] = task
    all_metrics['checkpoint'] = args.checkpoint
    
    # Save metrics JSON
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Saved metrics: {metrics_path}")
    
    # Save predictions CSV
    save_predictions_csv(predictions_clip, output_dir / 'predictions_clip.csv')
    
    # Confusion matrix
    print("\nGenerating confusion matrix...")
    
    cm_clip = confusion_matrix(clip_labels, clip_preds, labels=range(num_classes))
    
    plot_confusion_matrix(
        cm_clip, class_names,
        output_dir / 'confusion_matrix_clip.png',
        title=f'Confusion Matrix ({args.split}, {task})'
    )
    
    # Save raw confusion matrix
    np.save(output_dir / 'confusion_matrix_clip.npy', cm_clip)
    
    # Summary
    print(f"\n{'=' * 60}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Split: {args.split}")
    print(f"Task: {task}")
    print(f"Samples: {len(clip_labels)} clips")
    print()
    print("Overall Metrics:")
    print(f"  Top-1: {clip_metrics['clip_top1_acc']:.2f}%  Top-2: {clip_metrics['clip_top2_acc']:.2f}%")
    print(f"  Macro F1: {clip_metrics['clip_macro_f1']:.2f}%  Weighted F1: {clip_metrics['clip_weighted_f1']:.2f}%")
    print(f"  Macro Precision: {clip_metrics['clip_macro_precision']:.2f}%  Macro Recall: {clip_metrics['clip_macro_recall']:.2f}%")
    print(f"  Cohen's Kappa: {clip_metrics['clip_cohens_kappa']:.4f}")
    
    if task == 'tal':
        print()
        print("RMM vs Background (Binary):")
        print(f"  Accuracy: {clip_metrics['clip_rmm_vs_bg_accuracy']:.2f}%")
        print(f"  F1: {clip_metrics['clip_rmm_vs_bg_f1']:.2f}%  Precision: {clip_metrics['clip_rmm_vs_bg_precision']:.2f}%  Recall: {clip_metrics['clip_rmm_vs_bg_recall']:.2f}%")
        print(f"  AUC-ROC: {clip_metrics['clip_rmm_vs_bg_auc']:.2f}%")
        print(f"  BG False Alarm Rate: {clip_metrics['clip_bg_false_alarm_rate']:.2f}%  RMM Miss Rate: {clip_metrics['clip_rmm_miss_rate']:.2f}%")
        print()
        print("RMM-Only Metrics (excluding background):")
        print(f"  Macro F1: {clip_metrics['clip_rmm_only_macro_f1']:.2f}%")
        print(f"  Macro Precision: {clip_metrics['clip_rmm_only_macro_precision']:.2f}%  Macro Recall: {clip_metrics['clip_rmm_only_macro_recall']:.2f}%")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
