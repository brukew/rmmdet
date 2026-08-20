#!/usr/bin/env python3
"""
Analyze TAL training results across models and modalities.
Extracts per-class metrics, confusion patterns, and cross-model comparisons.
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re

# Resolve paths relative to the repo root so this runs from any clone location.
_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "paths.py").exists())

# Class names
CLASS_NAMES = ['hands flapping', 'jumping', 'rocking', 'spinning', 'background']
CLASS_SHORT = ['flapping', 'jumping', 'rocking', 'spinning', 'bg']


def extract_stgcn_metrics_from_log(log_path):
    """Extract final validation metrics from STGCN++ log file."""
    results = []
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Pattern to match Epoch(val) [20] lines
    pattern = r'Epoch\(val\) \[20\]\[\d+\]\s+top1_acc: ([\d.]+), top2_acc: ([\d.]+), mean_class_accuracy: ([\d.]+)'
    matches = re.findall(pattern, content)
    
    # Pattern to find which modality/fold by looking at feats=['x']
    feat_pattern = r"dict\(type='GenSkeFeat', dataset='coco', feats=\['([a-z]+)'\]\)"
    feat_matches = re.findall(feat_pattern, content)
    
    # Pattern to find fold by looking at ann_file
    fold_pattern = r"ann_file='data/sails/tal/cv_4class/5class_windows_conf04/fold(\d)\.pkl'"
    fold_matches = re.findall(fold_pattern, content)
    
    # Group results by modality and fold
    # Each config appears twice (once for each fold)
    # Order: j/fold0, j/fold1, b/fold0, b/fold1, jm/fold0, jm/fold1, bm/fold0, bm/fold1
    
    if len(matches) == 8 and len(feat_matches) >= 32 and len(fold_matches) >= 32:
        modalities = []
        folds = []
        
        # Extract modality and fold for each of the 8 training runs
        # Each run has 4 config sections (train/val/test pipelines + dataset)
        for i in range(8):
            # Get the modality from the first occurrence in this run's config
            mod_idx = i * 4
            if mod_idx < len(feat_matches):
                modalities.append(feat_matches[mod_idx])
            
            # Get the fold (appears in multiple places, use first)
            fold_idx = i * 4
            if fold_idx < len(fold_matches):
                folds.append(int(fold_matches[fold_idx]))
        
        for i, (top1, top2, mean_acc) in enumerate(matches):
            if i < len(modalities) and i < len(folds):
                results.append({
                    'modality': modalities[i],
                    'fold': folds[i],
                    'top1_acc': float(top1) * 100,
                    'top2_acc': float(top2) * 100,
                    'mean_class_acc': float(mean_acc) * 100
                })
    
    return results


def load_posec3d_metrics(base_dir):
    """Load PoseC3D metrics from JSON files."""
    base_path = Path(base_dir)
    results = []
    
    for fold in [0, 1]:
        metrics_file = base_path / f"fold{fold}/eval_val/metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                data = json.load(f)
            
            result = {
                'model': 'PoseC3D',
                'modality': 'joint',
                'fold': fold,
                'top1_acc': data['clip_top1_acc'],
                'top2_acc': data['clip_top2_acc'],
                'macro_f1': data['clip_macro_f1'],
                'weighted_f1': data['clip_weighted_f1'],
                'macro_precision': data['clip_macro_precision'],
                'macro_recall': data['clip_macro_recall'],
                'cohens_kappa': data['clip_cohens_kappa'],
            }
            
            # Per-class metrics (excluding background)
            for cls in ['hands flapping', 'jumping', 'rocking', 'spinning']:
                result[f'f1_{cls}'] = data[f'clip_f1_{cls}']
                result[f'prec_{cls}'] = data[f'clip_prec_{cls}']
                result[f'rec_{cls}'] = data[f'clip_rec_{cls}']
            
            results.append(result)
    
    return results


def load_stgcn_metrics(log_path, base_dir):
    """Load STGCN++ metrics from log and checkpoint dirs."""
    # Extract basic metrics from log
    log_results = extract_stgcn_metrics_from_log(log_path)
    
    # Enhance with directory structure info
    base_path = Path(base_dir)
    results = []
    
    for lr in log_results:
        result = {
            'model': 'STGCN++',
            'modality': lr['modality'],
            'fold': lr['fold'],
            'top1_acc': lr['top1_acc'],
            'top2_acc': lr['top2_acc'],
            'mean_class_acc': lr['mean_class_acc'],
        }
        results.append(result)
    
    return results


def create_summary_table(posec3d_results, stgcn_results):
    """Create a comprehensive summary table."""
    
    # PoseC3D summary
    print("\n" + "="*80)
    print("POSEC3D RESULTS (Joint Modality)")
    print("="*80)
    
    for r in posec3d_results:
        print(f"\nFold {r['fold']}:")
        print(f"  Overall Metrics:")
        print(f"    Top-1 Accuracy:    {r['top1_acc']:6.2f}%")
        print(f"    Top-2 Accuracy:    {r['top2_acc']:6.2f}%")
        print(f"    Macro F1:          {r['macro_f1']:6.2f}%")
        print(f"    Weighted F1:       {r['weighted_f1']:6.2f}%")
        print(f"    Cohen's Kappa:     {r['cohens_kappa']:6.4f}")
        print(f"\n  Per-Class Performance (F1 / Precision / Recall):")
        
        for cls in ['hands flapping', 'jumping', 'rocking', 'spinning']:
            f1 = r[f'f1_{cls}']
            prec = r[f'prec_{cls}']
            rec = r[f'rec_{cls}']
            print(f"    {cls:16s}: {f1:6.2f}% / {prec:6.2f}% / {rec:6.2f}%")
    
    # Average across folds
    if len(posec3d_results) == 2:
        print(f"\n  Average (Folds 0-1):")
        avg_top1 = np.mean([r['top1_acc'] for r in posec3d_results])
        avg_macro_f1 = np.mean([r['macro_f1'] for r in posec3d_results])
        print(f"    Top-1 Accuracy:    {avg_top1:6.2f}%")
        print(f"    Macro F1:          {avg_macro_f1:6.2f}%")
        
        print(f"\n    Per-Class F1 Averages:")
        for cls in ['hands flapping', 'jumping', 'rocking', 'spinning']:
            avg_f1 = np.mean([r[f'f1_{cls}'] for r in posec3d_results])
            print(f"      {cls:16s}: {avg_f1:6.2f}%")
    
    # STGCN++ summary
    print("\n" + "="*80)
    print("STGCN++ RESULTS (All Modalities)")
    print("="*80)
    
    # Group by modality
    by_modality = defaultdict(list)
    for r in stgcn_results:
        by_modality[r['modality']].append(r)
    
    modality_names = {'j': 'Joint', 'b': 'Bone', 'jm': 'Joint Motion', 'bm': 'Bone Motion'}
    
    for mod in ['j', 'b', 'jm', 'bm']:
        if mod in by_modality:
            print(f"\n{modality_names[mod]} Modality:")
            
            for r in sorted(by_modality[mod], key=lambda x: x['fold']):
                print(f"  Fold {r['fold']}:")
                print(f"    Top-1 Accuracy:      {r['top1_acc']:6.2f}%")
                print(f"    Top-2 Accuracy:      {r['top2_acc']:6.2f}%")
                print(f"    Mean Class Accuracy: {r['mean_class_acc']:6.2f}%")
            
            # Average
            if len(by_modality[mod]) == 2:
                avg_top1 = np.mean([r['top1_acc'] for r in by_modality[mod]])
                avg_mca = np.mean([r['mean_class_acc'] for r in by_modality[mod]])
                print(f"  Average (Folds 0-1):")
                print(f"    Top-1 Accuracy:      {avg_top1:6.2f}%")
                print(f"    Mean Class Accuracy: {avg_mca:6.2f}%")
    
    # Cross-modality comparison
    print("\n" + "="*80)
    print("CROSS-MODALITY COMPARISON (STGCN++, Averaged over Folds 0-1)")
    print("="*80)
    
    comparison = []
    for mod in ['j', 'b', 'jm', 'bm']:
        if mod in by_modality and len(by_modality[mod]) == 2:
            avg_top1 = np.mean([r['top1_acc'] for r in by_modality[mod]])
            avg_top2 = np.mean([r['top2_acc'] for r in by_modality[mod]])
            avg_mca = np.mean([r['mean_class_acc'] for r in by_modality[mod]])
            comparison.append({
                'Modality': modality_names[mod],
                'Top-1 Acc': f"{avg_top1:.2f}%",
                'Top-2 Acc': f"{avg_top2:.2f}%",
                'Mean Class Acc': f"{avg_mca:.2f}%"
            })
    
    df_comp = pd.DataFrame(comparison)
    print(df_comp.to_string(index=False))
    
    # Model comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON (Averaged over Folds 0-1)")
    print("="*80)
    
    model_comp = []
    
    # PoseC3D
    if len(posec3d_results) == 2:
        model_comp.append({
            'Model': 'PoseC3D (joint)',
            'Top-1 Acc': f"{np.mean([r['top1_acc'] for r in posec3d_results]):.2f}%",
            'Macro F1': f"{np.mean([r['macro_f1'] for r in posec3d_results]):.2f}%",
            'Cohen\'s κ': f"{np.mean([r['cohens_kappa'] for r in posec3d_results]):.4f}"
        })
    
    # STGCN++ best modality
    best_mod = None
    best_acc = 0
    for mod in ['j', 'b', 'jm', 'bm']:
        if mod in by_modality and len(by_modality[mod]) == 2:
            avg_top1 = np.mean([r['top1_acc'] for r in by_modality[mod]])
            if avg_top1 > best_acc:
                best_acc = avg_top1
                best_mod = mod
    
    if best_mod:
        model_comp.append({
            'Model': f'STGCN++ ({modality_names[best_mod].lower()})',
            'Top-1 Acc': f"{best_acc:.2f}%",
            'Macro F1': 'N/A',
            'Cohen\'s κ': 'N/A'
        })
    
    df_model = pd.DataFrame(model_comp)
    print(df_model.to_string(index=False))
    
    # Key observations
    print("\n" + "="*80)
    print("KEY OBSERVATIONS")
    print("="*80)
    
    print("\n1. Class Imbalance Impact (PoseC3D):")
    if len(posec3d_results) >= 1:
        r = posec3d_results[0]
        print(f"   - 'Rocking' class: {r['f1_rocking']:.2f}% F1 (FAILED TO LEARN)")
        print(f"   - 'Spinning' class: {r['f1_spinning']:.2f}% F1 (POOR)")
        print(f"   - 'Hands flapping': {r['f1_hands flapping']:.2f}% F1 (BEST)")
        print(f"   - 'Jumping': {r['f1_jumping']:.2f}% F1 (MODERATE)")
        print("   → Despite background subsampling, minority classes still struggle")
    
    print("\n2. STGCN++ Modality Performance:")
    if len(by_modality) == 4:
        mod_accs = {mod: np.mean([r['top1_acc'] for r in by_modality[mod]]) 
                    for mod in ['j', 'b', 'jm', 'bm'] if len(by_modality[mod]) == 2}
        best = max(mod_accs.items(), key=lambda x: x[1])
        worst = min(mod_accs.items(), key=lambda x: x[1])
        print(f"   - Best:  {modality_names[best[0]]:15s} {best[1]:.2f}%")
        print(f"   - Worst: {modality_names[worst[0]]:15s} {worst[1]:.2f}%")
        print(f"   - Spread: {best[1] - worst[1]:.2f}%")
    
    print("\n3. Top-1 vs Mean Class Accuracy Gap (STGCN++):")
    for mod in ['j', 'b', 'jm', 'bm']:
        if mod in by_modality and len(by_modality[mod]) == 2:
            avg_top1 = np.mean([r['top1_acc'] for r in by_modality[mod]])
            avg_mca = np.mean([r['mean_class_acc'] for r in by_modality[mod]])
            gap = avg_top1 - avg_mca
            print(f"   - {modality_names[mod]:15s}: {gap:.2f}% gap")
    print("   → Large gap indicates model relies heavily on background class")
    
    print("\n4. Model Architecture Comparison:")
    if len(posec3d_results) == 2 and 'j' in by_modality:
        posec3d_avg = np.mean([r['top1_acc'] for r in posec3d_results])
        stgcn_j_avg = np.mean([r['top1_acc'] for r in by_modality['j']])
        diff = posec3d_avg - stgcn_j_avg
        print(f"   - PoseC3D (joint):  {posec3d_avg:.2f}%")
        print(f"   - STGCN++ (joint):  {stgcn_j_avg:.2f}%")
        print(f"   - Difference:       {diff:+.2f}%")
        if abs(diff) < 5:
            print("   → Models perform similarly on joint modality")
        elif diff > 0:
            print("   → PoseC3D outperforms STGCN++ on joint modality")
        else:
            print("   → STGCN++ outperforms PoseC3D on joint modality")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--posec3d-dir",
        default=str(_REPO_ROOT / "pyskl/work_dirs/posec3d/tal/cv_4class_5class_bgsub"),
        help="PoseC3D TAL work_dir containing per-fold metrics.",
    )
    parser.add_argument(
        "--stgcn-dir",
        default=str(_REPO_ROOT / "pyskl/work_dirs/stgcnpp/tal/cv_4class_5class_bgsub"),
        help="ST-GCN++ TAL work_dir containing per-fold metrics.",
    )
    parser.add_argument(
        "--stgcn-log",
        required=True,
        help="ST-GCN++ TAL training log (.err) to scrape per-class metrics from. "
             "This is a SLURM job artifact and is not shipped with the repo; pass "
             "the log from your own training run (see slurm-logs/).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    posec3d_dir = args.posec3d_dir
    stgcn_log = args.stgcn_log
    stgcn_dir = args.stgcn_dir
    
    # Load results
    print("Loading results...")
    posec3d_results = load_posec3d_metrics(posec3d_dir)
    stgcn_results = load_stgcn_metrics(stgcn_log, stgcn_dir)
    
    print(f"Loaded {len(posec3d_results)} PoseC3D results")
    print(f"Loaded {len(stgcn_results)} STGCN++ results")
    
    # Create summary
    create_summary_table(posec3d_results, stgcn_results)
    
    # Save to JSON
    output = {
        'posec3d': posec3d_results,
        'stgcn++': stgcn_results,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    output_file = Path(posec3d_dir).parent / "tal_results_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()


