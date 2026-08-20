#!/usr/bin/env python3
"""
Grid search over TAL postprocessing parameters.

Sweeps over threshold, smoothing window, and merge gap to find optimal
settings for each model. Uses existing TAL-format prediction CSVs.

Usage:
    python scripts/grid_search_postprocessing.py
    
    # With custom output directory
    python scripts/grid_search_postprocessing.py --out-dir grid_search_output
    
    # Single model only
    python scripts/grid_search_postprocessing.py --models vjepa
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from window_to_segments import PostprocessParams, window_scores_to_segments
from tal_map_eval import compute_map, load_gt_segments, ID2LABEL_4CLASS


@dataclass
class ModelConfig:
    """Configuration for a model to evaluate."""
    name: str
    pred_csv_pattern: str  # Pattern with {fold} placeholder
    folds: List[int]


# Model configurations
MODELS = {
    "posec3d_ce": ModelConfig(
        name="PoseC3D (CE)",
        pred_csv_pattern="posec3d_ce/fold{fold}/tal_format_preds.csv",
        folds=[0, 1],
    ),
    "posec3d_focal": ModelConfig(
        name="PoseC3D (Focal)",
        pred_csv_pattern="posec3d_focal/fold{fold}/tal_format_preds.csv",
        folds=[0, 1, 2],
    ),
    "stgcnpp_ce_4stream": ModelConfig(
        name="STGCN++ 4-Stream (CE)",
        pred_csv_pattern="stgcnpp_ce_4stream/fold{fold}/fused_tal_format_preds.csv",
        folds=[0, 1],
    ),
    "vjepa": ModelConfig(
        name="V-JEPA",
        pred_csv_pattern="vjepa/fold{fold}/tal_format_preds.csv",
        folds=[0, 1],
    ),
    # Fusion models (V-JEPA + skeleton via MLP on log-probs)
    "vjepa_posec3d_mlp_logp": ModelConfig(
        name="V-JEPA + PoseC3D (MLP)",
        pred_csv_pattern="vjepa_posec3d_mlp_logp/fold{fold}/tal_format_preds.csv",
        folds=[0, 1],
    ),
    "vjepa_stgcnpp_mlp_logp": ModelConfig(
        name="V-JEPA + STGCN++ (MLP)",
        pred_csv_pattern="vjepa_stgcnpp_mlp_logp/fold{fold}/tal_format_preds.csv",
        folds=[0, 1],
    ),
    # PoseC3D with balanced sampling
    "posec3d_ce_balanced": ModelConfig(
        name="PoseC3D (CE Balanced)",
        pred_csv_pattern="posec3d_ce_balanced/fold{fold}/tal_format_preds.csv",
        folds=[0, 1, 2],
    ),
    # Fusion: V-JEPA + PoseC3D Balanced
    "vjepa_posec3d_bal_mlp_logp": ModelConfig(
        name="V-JEPA + PoseC3D Bal (MLP)",
        pred_csv_pattern="vjepa_posec3d_bal_mlp_logp/fold{fold}/tal_format_preds.csv",
        folds=[0, 1],
    ),
    # Fusion: V-JEPA Balanced + PoseC3D CE
    "vjepa_balanced_posec3d_mlp_logp": ModelConfig(
        name="V-JEPA Balanced + PoseC3D (CE)",
        pred_csv_pattern="vjepa_balanced_posec3d_mlp_logp/fold{fold}/tal_format_preds.csv",
        folds=[0, 1],
    ),
}

# Parameter grid
PARAM_GRID = {
    "thr": [0.3, 0.4, 0.5, 0.6],
    "smooth_k": [1, 3, 5, 7],
    "merge_gap_sec": [0.5, 1.0, 1.5, 2.0],
}

# Fixed parameters
TIOU_THRESHOLDS = [0.3, 0.5, 0.7]
CLASS_IDS = [0, 1, 2, 3]  # 4-class TAL (excluding background)


def evaluate_single_fold(
    pred_csv: Path,
    gt_by_video: Dict,
    params: PostprocessParams,
) -> Dict[str, float]:
    """
    Evaluate a single fold with given postprocessing parameters.
    
    Returns:
        Dict with mAP@0.3, mAP@0.5, mAP@0.7, avg_mAP
    """
    # Load predictions
    df = pd.read_csv(pred_csv)
    
    # Postprocess to segments
    pred_segments = window_scores_to_segments(df, params)
    
    if pred_segments.empty:
        return {"mAP@0.3": 0.0, "mAP@0.5": 0.0, "mAP@0.7": 0.0, "avg_mAP": 0.0}
    
    # Compute mAP
    metrics = compute_map(
        pred_segments,
        gt_by_video,
        CLASS_IDS,
        TIOU_THRESHOLDS,
    )
    
    return {
        "mAP@0.3": metrics.get("mAP@0.3", 0.0),
        "mAP@0.5": metrics.get("mAP@0.5", 0.0),
        "mAP@0.7": metrics.get("mAP@0.7", 0.0),
        "avg_mAP": metrics.get("avg_mAP", 0.0),
    }


def run_grid_search(
    eval_results_dir: Path,
    splits_root: Path,
    models_to_run: Optional[List[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run grid search over all models and parameter combinations.
    
    Args:
        eval_results_dir: Directory containing TAL-format prediction CSVs
        splits_root: Root directory for GT segment CSVs
        models_to_run: Optional list of model keys to run (default: all)
        verbose: Print progress
        
    Returns:
        DataFrame with all results
    """
    # Load GT segments once
    if verbose:
        print("Loading GT segments...")
    gt_by_video = load_gt_segments(splits_root, task="4class", mode="cv")
    if verbose:
        print(f"  Loaded {sum(len(v) for v in gt_by_video.values())} segments from {len(gt_by_video)} videos")
    
    # Generate all parameter combinations
    param_keys = list(PARAM_GRID.keys())
    param_values = [PARAM_GRID[k] for k in param_keys]
    param_combinations = list(product(*param_values))
    
    if verbose:
        print(f"\nParameter grid: {len(param_combinations)} combinations")
        for k, v in PARAM_GRID.items():
            print(f"  {k}: {v}")
    
    # Results storage
    results = []
    
    # Select models to run
    if models_to_run is None:
        models_to_run = list(MODELS.keys())
    
    # Run grid search for each model
    for model_key in models_to_run:
        if model_key not in MODELS:
            print(f"Warning: Unknown model '{model_key}', skipping")
            continue
            
        config = MODELS[model_key]
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Model: {config.name}")
            print(f"{'='*60}")
        
        # Check which folds exist
        available_folds = []
        for fold in config.folds:
            pred_csv = eval_results_dir / config.pred_csv_pattern.format(fold=fold)
            if pred_csv.exists():
                available_folds.append(fold)
            else:
                if verbose:
                    print(f"  Warning: Missing fold {fold}: {pred_csv}")
        
        if not available_folds:
            if verbose:
                print(f"  No folds available, skipping model")
            continue
        
        if verbose:
            print(f"  Folds: {available_folds}")
            print(f"  Running {len(param_combinations)} parameter combinations...")
        
        # Run each parameter combination
        for i, param_vals in enumerate(param_combinations):
            params_dict = dict(zip(param_keys, param_vals))
            
            # Create PostprocessParams
            pp_params = PostprocessParams(
                smooth_k=params_dict["smooth_k"],
                threshold=params_dict["thr"],
                merge_gap_sec=params_dict["merge_gap_sec"],
                min_duration_sec=0.0,
                score_reducer="max",
                class_ids=CLASS_IDS,
            )
            
            # Evaluate each fold
            fold_metrics = []
            for fold in available_folds:
                pred_csv = eval_results_dir / config.pred_csv_pattern.format(fold=fold)
                metrics = evaluate_single_fold(pred_csv, gt_by_video, pp_params)
                fold_metrics.append(metrics)
            
            # Average across folds
            avg_metrics = {
                k: np.mean([m[k] for m in fold_metrics])
                for k in fold_metrics[0].keys()
            }
            std_metrics = {
                f"{k}_std": np.std([m[k] for m in fold_metrics])
                for k in fold_metrics[0].keys()
            }
            
            # Store result
            result = {
                "model": model_key,
                "model_name": config.name,
                "n_folds": len(available_folds),
                **params_dict,
                **avg_metrics,
                **std_metrics,
            }
            results.append(result)
            
            # Progress
            if verbose and (i + 1) % 16 == 0:
                print(f"    {i + 1}/{len(param_combinations)} combinations done...")
        
        if verbose:
            # Find best for this model
            model_results = [r for r in results if r["model"] == model_key]
            best = max(model_results, key=lambda x: x["avg_mAP"])
            print(f"\n  Best for {config.name}:")
            print(f"    thr={best['thr']}, smooth_k={best['smooth_k']}, merge_gap={best['merge_gap_sec']}")
            print(f"    mAP@0.5: {best['mAP@0.5']:.4f}, avg_mAP: {best['avg_mAP']:.4f}")
    
    return pd.DataFrame(results)


def find_best_params(results_df: pd.DataFrame) -> Dict:
    """
    Find best parameters for each model based on avg_mAP.
    
    Returns:
        Dict mapping model -> best params and metrics
    """
    best_params = {}
    
    for model_key in results_df["model"].unique():
        model_df = results_df[results_df["model"] == model_key]
        best_row = model_df.loc[model_df["avg_mAP"].idxmax()]
        
        best_params[model_key] = {
            "model_name": best_row["model_name"],
            "best_params": {
                "thr": best_row["thr"],
                "smooth_k": int(best_row["smooth_k"]),
                "merge_gap_sec": best_row["merge_gap_sec"],
            },
            "metrics": {
                "mAP@0.3": best_row["mAP@0.3"],
                "mAP@0.5": best_row["mAP@0.5"],
                "mAP@0.7": best_row["mAP@0.7"],
                "avg_mAP": best_row["avg_mAP"],
            },
            "improvement_vs_default": None,  # Will compute if default exists
        }
    
    return best_params


def main():
    parser = argparse.ArgumentParser(
        description="Grid search over TAL postprocessing parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--eval-results-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "tal" / "eval_results",
        help="Directory containing TAL-format prediction CSVs",
    )
    parser.add_argument(
        "--splits-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataprep" / "splits",
        help="Root directory for GT segment CSVs",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: eval_results_dir)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to run (default: all)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    
    args = parser.parse_args()
    
    out_dir = args.out_dir or args.eval_results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("TAL Postprocessing Grid Search")
    print("="*60)
    print(f"Eval results dir: {args.eval_results_dir}")
    print(f"Splits root: {args.splits_root}")
    print(f"Output dir: {out_dir}")
    
    # Run grid search
    results_df = run_grid_search(
        args.eval_results_dir,
        args.splits_root,
        models_to_run=args.models,
        verbose=not args.quiet,
    )
    
    if results_df.empty:
        print("\nNo results generated. Check that prediction CSVs exist.")
        return
    
    # Save all results
    results_csv = out_dir / "grid_search_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\nAll results saved to: {results_csv}")
    
    # Find and save best params
    best_params = find_best_params(results_df)
    best_params_json = out_dir / "best_params_per_model.json"
    with open(best_params_json, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"Best params saved to: {best_params_json}")
    
    # Print summary
    print("\n" + "="*60)
    print("BEST PARAMETERS PER MODEL")
    print("="*60)
    
    # Compare with default (thr=0.5, smooth_k=3, merge_gap=1.0)
    default_params = {"thr": 0.5, "smooth_k": 3, "merge_gap_sec": 1.0}
    
    for model_key, data in best_params.items():
        print(f"\n{data['model_name']}:")
        print(f"  Best params: thr={data['best_params']['thr']}, "
              f"smooth_k={data['best_params']['smooth_k']}, "
              f"merge_gap={data['best_params']['merge_gap_sec']}")
        print(f"  Best mAP@0.5: {data['metrics']['mAP@0.5']:.4f}")
        print(f"  Best avg_mAP: {data['metrics']['avg_mAP']:.4f}")
        
        # Find default performance
        default_row = results_df[
            (results_df["model"] == model_key) &
            (results_df["thr"] == default_params["thr"]) &
            (results_df["smooth_k"] == default_params["smooth_k"]) &
            (results_df["merge_gap_sec"] == default_params["merge_gap_sec"])
        ]
        
        if not default_row.empty:
            default_avg_map = default_row["avg_mAP"].values[0]
            improvement = data["metrics"]["avg_mAP"] - default_avg_map
            pct_improvement = (improvement / default_avg_map * 100) if default_avg_map > 0 else 0
            print(f"  Default avg_mAP: {default_avg_map:.4f}")
            print(f"  Improvement: +{improvement:.4f} ({pct_improvement:+.1f}%)")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()

