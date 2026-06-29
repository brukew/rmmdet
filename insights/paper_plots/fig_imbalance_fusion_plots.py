#!/usr/bin/env python3
"""
Generate comparison plots for class imbalance remedies and fusion strategies.

This script generates 3 plots:
1. Classification Task Imbalance Remedies (Small Multiples) - PoseC3D & STGCN++
2. Window Detection Task Imbalance Remedies (Faceted Bars) - PoseC3D, STGCN++, V-JEPA
3. Fusion Strategy Comparison (Grouped Bars) - 4-class classification

Data sources:
- MODEL_COMPARISON.md (classification task)
- TAL_DET_MODEL_COMPARISON.md (window detection task)
"""

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Color Palette (consistent with existing project plots)
# =============================================================================
COLORS = {
    'PoseC3D': '#FFB74D',      # Orange
    'STGCN++': '#81C784',      # Green
    'V-JEPA': '#BA68C8',       # Purple
    'fusion_light': '#90CAF9', # Light blue
    'fusion_mid': '#64B5F6',   # Medium blue
    'fusion_dark': '#2196F3',  # Dark blue
}

# Metric colors for grouped bars
METRIC_COLORS = {
    'Top-1 Acc': '#64B5F6',    # Blue
    'Macro F1': '#FFB74D',     # Orange
}


# =============================================================================
# Data Definitions (extracted from markdown files)
# =============================================================================

# Plot 1: Classification Task - 5-class Imbalance Remedies
CLASSIFICATION_DATA = {
    'PoseC3D': {
        'variants': ['Weighted', 'Weighted Sqrt', 'Focal Loss'],
        'macro_f1': [65.9, 63.9, 61.8],
        'macro_f1_std': [2.8, 3.2, 3.5],
    },
    'STGCN++': {
        'variants': ['Non-weighted', 'Weighted', 'Focal'],
        'macro_f1': [64.6, 63.7, 63.6],
        'macro_f1_std': [1.4, 2.2, 2.9],
    },
}

# Plot 2: Window Detection Task - 5-class Imbalance Remedies
WINDOW_DETECTION_DATA = {
    'PoseC3D': {
        'variants': ['CE (Default)', 'CE Balanced', 'Focal'],
        'top1_acc': [85.89, 81.11, 84.31],
        'top1_acc_std': [0.56, 1.67, 1.33],
        'macro_f1': [37.62, 35.82, 31.47],
        'macro_f1_std': [3.23, 5.78, 1.80],
    },
    'STGCN++': {
        'variants': ['CE (Default)', 'CE Balanced', 'Focal'],
        'top1_acc': [85.74, 77.48, 37.13],
        'top1_acc_std': [0.14, 3.75, 3.75],
        'macro_f1': [32.85, 30.40, 19.56],
        'macro_f1_std': [1.48, 1.35, 1.18],
    },
    'V-JEPA': {
        'variants': ['Default', 'Balanced'],
        'top1_acc': [81.75, 80.94],
        'top1_acc_std': [3.28, 1.73],
        'macro_f1': [36.10, 36.83],
        'macro_f1_std': [3.23, 3.45],
    },
}

# Plot 3: Fusion Strategy Comparison - 4-class Classification
FUSION_DATA = {
    'strategies': ['Scalar α', 'Per-class α', 'MLP'],
    'clip_top1': [82.6, 82.6, 84.2],
    'clip_top1_std': [3.0, 3.6, 4.8],
    'clip_macro_f1': [81.7, 81.2, 84.9],
    'clip_macro_f1_std': [3.2, 2.7, 4.0],
}


# =============================================================================
# Plot 1: Classification Imbalance - Small Multiples
# =============================================================================
def plot_classification_imbalance(output_dir: Path) -> dict:
    """
    Generate small multiples bar chart for classification task imbalance remedies.
    
    Creates a 1x2 figure with separate panels for PoseC3D and STGCN++.
    
    Args:
        output_dir: Directory to save the output figure
        
    Returns:
        Dictionary containing the plot data for JSON export
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    fig.suptitle('5-Class Classification: Imbalance Remedy Comparison', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    models = ['PoseC3D', 'STGCN++']
    
    for idx, (ax, model) in enumerate(zip(axes, models)):
        data = CLASSIFICATION_DATA[model]
        variants = data['variants']
        values = data['macro_f1']
        errors = data['macro_f1_std']
        
        x = np.arange(len(variants))
        bars = ax.bar(x, values, yerr=errors, capsize=4, 
                      color=COLORS[model], edgecolor='black', linewidth=0.5,
                      alpha=0.85, error_kw={'linewidth': 1.5})
        
        # Add value labels on bars (positioned below error bars)
        for bar, val, err in zip(bars, values, errors):
            height = bar.get_height()
            label_y = height - err  # Position below the error bar
            ax.annotate(f'{val:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, label_y),
                        xytext=(0, -3),
                        textcoords="offset points",
                        ha='center', va='top', fontsize=9, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=15, ha='right', fontsize=10)
        ax.set_title(f'{model}', fontsize=12, fontweight='bold', color=COLORS[model])
        ax.set_ylim(55, 72)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if idx == 0:
            ax.set_ylabel('Macro F1 (%)', fontsize=11)
    
    plt.tight_layout()
    
    # Save figure
    fig.savefig(output_dir / 'fig_class_imbalance_classification.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / 'fig_class_imbalance_classification.pdf', 
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  Saved: fig_class_imbalance_classification.png/pdf")
    
    return CLASSIFICATION_DATA


# =============================================================================
# Plot 2: Window Detection Imbalance - Faceted Bars
# =============================================================================
def plot_window_detection_imbalance(output_dir: Path) -> dict:
    """
    Generate faceted bar chart for window detection task imbalance remedies.
    
    Creates a 1x3 figure with panels for PoseC3D, STGCN++, and V-JEPA.
    Each panel shows Top-1 Accuracy and Macro F1 as grouped bars.
    
    Args:
        output_dir: Directory to save the output figure
        
    Returns:
        Dictionary containing the plot data for JSON export
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Window-Level Detection: Imbalance Remedy Comparison', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    models = ['PoseC3D', 'STGCN++', 'V-JEPA']
    
    for idx, (ax, model) in enumerate(zip(axes, models)):
        data = WINDOW_DETECTION_DATA[model]
        variants = data['variants']
        n_variants = len(variants)
        
        x = np.arange(n_variants)
        width = 0.35
        
        # Top-1 Accuracy bars
        bars1 = ax.bar(x - width/2, data['top1_acc'], width, 
                       yerr=data['top1_acc_std'], capsize=3,
                       label='Top-1 Acc', color=METRIC_COLORS['Top-1 Acc'],
                       edgecolor='black', linewidth=0.5, alpha=0.85,
                       error_kw={'linewidth': 1.2})
        
        # Macro F1 bars
        bars2 = ax.bar(x + width/2, data['macro_f1'], width,
                       yerr=data['macro_f1_std'], capsize=3,
                       label='Macro F1', color=METRIC_COLORS['Macro F1'],
                       edgecolor='black', linewidth=0.5, alpha=0.85,
                       error_kw={'linewidth': 1.2})
        
        # Add value labels on bars (positioned below error bars)
        for bars, vals, errs in [(bars1, data['top1_acc'], data['top1_acc_std']), 
                                  (bars2, data['macro_f1'], data['macro_f1_std'])]:
            for bar, val, err in zip(bars, vals, errs):
                height = bar.get_height()
                label_y = height - err  # Position below the error bar
                ax.annotate(f'{val:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, label_y),
                            xytext=(0, -2),
                            textcoords="offset points",
                            ha='center', va='top', fontsize=7, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=20, ha='right', fontsize=9)
        ax.set_title(f'{model}', fontsize=12, fontweight='bold', color=COLORS[model])
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if idx == 0:
            ax.set_ylabel('Metric Value (%)', fontsize=11)
        if idx == 1:
            ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    fig.savefig(output_dir / 'fig_class_imbalance_window_detection.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / 'fig_class_imbalance_window_detection.pdf', 
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  Saved: fig_class_imbalance_window_detection.png/pdf")
    
    return WINDOW_DETECTION_DATA


# =============================================================================
# Plot 3: Fusion Strategy Comparison - Grouped Bars
# =============================================================================
def plot_fusion_strategies(output_dir: Path) -> dict:
    """
    Generate grouped bar chart for fusion strategy comparison.
    
    Shows Clip Top-1 Accuracy and Clip Macro F1 for each fusion strategy.
    
    Args:
        output_dir: Directory to save the output figure
        
    Returns:
        Dictionary containing the plot data for JSON export
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    strategies = FUSION_DATA['strategies']
    n_strategies = len(strategies)
    
    x = np.arange(n_strategies)
    width = 0.35
    
    # Clip Top-1 Accuracy bars
    bars1 = ax.bar(x - width/2, FUSION_DATA['clip_top1'], width,
                   yerr=FUSION_DATA['clip_top1_std'], capsize=5,
                   label='Clip Top-1 Acc', color=METRIC_COLORS['Top-1 Acc'],
                   edgecolor='black', linewidth=0.5, alpha=0.85,
                   error_kw={'linewidth': 1.5})
    
    # Clip Macro F1 bars
    bars2 = ax.bar(x + width/2, FUSION_DATA['clip_macro_f1'], width,
                   yerr=FUSION_DATA['clip_macro_f1_std'], capsize=5,
                   label='Clip Macro F1', color=METRIC_COLORS['Macro F1'],
                   edgecolor='black', linewidth=0.5, alpha=0.85,
                   error_kw={'linewidth': 1.5})
    
    # Add value labels on bars (positioned below error bars)
    for bars, vals, errs in [(bars1, FUSION_DATA['clip_top1'], FUSION_DATA['clip_top1_std']), 
                              (bars2, FUSION_DATA['clip_macro_f1'], FUSION_DATA['clip_macro_f1_std'])]:
        for bar, val, err in zip(bars, vals, errs):
            height = bar.get_height()
            label_y = height - err  # Position below the error bar
            ax.annotate(f'{val:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, label_y),
                        xytext=(0, -3),
                        textcoords="offset points",
                        ha='center', va='top', fontsize=10, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=11)
    ax.set_ylabel('Metric Value (%)', fontsize=11)
    ax.set_title('4-Class Fusion Strategy Comparison (V-JEPA + PoseC3D)', 
                 fontsize=13, fontweight='bold')
    ax.set_ylim(75, 92)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    fig.savefig(output_dir / 'fig_fusion_strategy_comparison.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / 'fig_fusion_strategy_comparison.pdf', 
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  Saved: fig_fusion_strategy_comparison.png/pdf")
    
    return FUSION_DATA


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    """Main function to generate all plots and save data."""
    parser = argparse.ArgumentParser(
        description='Generate imbalance remedy and fusion strategy comparison plots'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='/orcd/data/satra/001/users/brukew/actreg/insights/figs',
        help='Output directory for figures'
    )
    parser.add_argument(
        '--tables-dir',
        type=str,
        default='/orcd/data/satra/001/users/brukew/actreg/insights/tables',
        help='Output directory for JSON data'
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    tables_dir = Path(args.tables_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Generating Imbalance and Fusion Comparison Plots")
    print("=" * 60)
    
    # Collect all data for JSON export
    all_data = {}
    
    # Plot 1: Classification Imbalance
    print("\n[1/3] Classification Task Imbalance Remedies (Small Multiples)")
    all_data['classification_imbalance'] = plot_classification_imbalance(output_dir)
    
    # Plot 2: Window Detection Imbalance
    print("\n[2/3] Window Detection Task Imbalance Remedies (Faceted Bars)")
    all_data['window_detection_imbalance'] = plot_window_detection_imbalance(output_dir)
    
    # Plot 3: Fusion Strategies
    print("\n[3/3] Fusion Strategy Comparison (Grouped Bars)")
    all_data['fusion_strategies'] = plot_fusion_strategies(output_dir)
    
    # Save all data to JSON
    json_path = tables_dir / 'fig_imbalance_fusion_data.json'
    with open(json_path, 'w') as f:
        json.dump(all_data, f, indent=2)
    print(f"\n  Saved data: {json_path}")
    
    print("\n" + "=" * 60)
    print("All plots generated successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
