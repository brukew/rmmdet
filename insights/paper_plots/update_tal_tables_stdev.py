#!/usr/bin/env python3
"""
Update TAL tables with per-class AP standard deviations.

This script computes per-class AP stdev from per-fold data for:
- Table 5.7: Segment-level TAL mAP (add per-class breakdown)
- Table 5.8: ActionFormer per-class AP (already has stdev, verify)
- Table 5.9: Binary TAL (add stdev where missing)

Usage:
    python update_tal_tables_stdev.py
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

# Project root
ACTREG_ROOT = Path(__file__).parent.parent.parent

# Paths
TABLES_DIR = ACTREG_ROOT / "insights" / "tables"
TAL_EVAL_DIR = ACTREG_ROOT / "tal" / "eval_results"

# Class names for 4-class TAL
CLASS_NAMES_4CLASS = {
    "0": "hands flapping",
    "1": "jumping",
    "2": "rocking",
    "3": "spinning"
}


def load_fold_metrics(model_dir: Path, fold: int, use_best_eval: bool = True) -> Optional[Dict]:
    """Load metrics for a single fold."""
    # Try different path patterns based on use_best_eval preference
    if use_best_eval:
        patterns = [
            model_dir / f"fold{fold}" / "best_eval" / "metrics.json",
            model_dir / f"fold_{fold}" / "best_eval" / "metrics.json",
            model_dir / f"fold{fold}" / "metrics.json",
            model_dir / f"fold_{fold}" / "metrics.json",
        ]
    else:
        patterns = [
            model_dir / f"fold{fold}" / "metrics.json",
            model_dir / f"fold_{fold}" / "metrics.json",
            model_dir / f"fold{fold}" / "best_eval" / "metrics.json",
            model_dir / f"fold_{fold}" / "best_eval" / "metrics.json",
        ]
    
    for path in patterns:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    
    return None


def compute_per_class_ap_stats(
    model_dir: Path,
    n_folds: int = 3,
    use_best_eval: bool = True
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compute per-class AP mean/std across folds.
    
    Returns:
        Dict mapping tIoU -> class_name -> {mean, std}
    """
    fold_data = []
    
    for fold in range(n_folds):
        metrics = load_fold_metrics(model_dir, fold, use_best_eval)
        if metrics and "per_class_ap" in metrics:
            fold_data.append(metrics["per_class_ap"])
    
    if not fold_data:
        return {}
    
    # Aggregate across folds
    results = {}
    
    # Get all tIoU thresholds
    tiou_keys = set()
    for fd in fold_data:
        tiou_keys.update(fd.keys())
    
    for tiou in sorted(tiou_keys):
        results[tiou] = {}
        
        # Get all classes
        class_keys = set()
        for fd in fold_data:
            if tiou in fd:
                class_keys.update(fd[tiou].keys())
        
        for class_key in sorted(class_keys):
            values = []
            for fd in fold_data:
                if tiou in fd and class_key in fd[tiou]:
                    values.append(fd[tiou][class_key])
            
            if values:
                class_name = CLASS_NAMES_4CLASS.get(class_key, class_key)
                results[tiou][class_name] = {
                    "mean": round(float(np.mean(values)) * 100, 2),
                    "std": round(float(np.std(values)) * 100, 2)
                }
    
    return results


def compute_mAP_stats(model_dir: Path, n_folds: int = 3, use_best_eval: bool = True) -> Dict[str, Dict[str, float]]:
    """
    Compute mAP mean/std across folds for different tIoU thresholds.
    """
    fold_data = []
    
    for fold in range(n_folds):
        metrics = load_fold_metrics(model_dir, fold, use_best_eval)
        if metrics:
            fold_data.append(metrics)
    
    if not fold_data:
        return {}
    
    results = {}
    
    # Get all mAP keys
    map_keys = [k for k in fold_data[0].keys() if k.startswith("mAP@") or k == "avg_mAP"]
    
    for key in map_keys:
        values = [fd[key] for fd in fold_data if key in fd]
        if values:
            results[key] = {
                "mean": round(float(np.mean(values)) * 100, 2),
                "std": round(float(np.std(values)) * 100, 2)
            }
    
    return results


def update_table_5_7():
    """Update Table 5.7 with per-class AP breakdown and add single modality models."""
    print("\n=== Updating Table 5.7 (Segment-level TAL mAP) ===")
    
    table_path = TABLES_DIR / "table_5_7_segment_tal_map.json"
    
    with open(table_path) as f:
        table = json.load(f)
    
    # All models to process (existing + single modality)
    # Using post-HPO (balanced) versions where available
    models_config = {
        # Existing fusion models
        "V-JEPA + PoseC3D (MLP)": {
            "dir": TAL_EVAL_DIR / "vjepa_posec3d_mlp_logp",
            "n_folds": 2,
            "use_best_eval": True
        },
        "V-JEPA + STGCN++ (MLP)": {
            "dir": TAL_EVAL_DIR / "vjepa_stgcnpp_mlp_logp",
            "n_folds": 2,
            "use_best_eval": True
        },
        "ActionFormer + V-JEPA Balanced": {
            "dir": TAL_EVAL_DIR / "actionformer_vjepa",
            "n_folds": 2,
            "use_best_eval": False  # Uses direct metrics.json
        },
        # Single modality models (post-HPO balanced versions)
        "V-JEPA": {
            "dir": TAL_EVAL_DIR / "vjepa_balanced",
            "n_folds": 3,
            "use_best_eval": True,
            "note": "post-HPO"
        },
        "PoseC3D": {
            "dir": TAL_EVAL_DIR / "posec3d_ce_balanced",
            "n_folds": 3,
            "use_best_eval": True,
            "note": "post-HPO"
        },
        "STGCN++": {
            "dir": TAL_EVAL_DIR / "stgcnpp_ce_4stream",
            "n_folds": 2,
            "use_best_eval": False,
            "note": "pre-HPO (no balanced version available)"
        },
    }
    
    # Update existing models
    for model_entry in table["models"]:
        model_name = model_entry["model"]
        
        if model_name in models_config:
            config = models_config[model_name]
            use_best_eval = config.get("use_best_eval", True)
            print(f"  Processing {model_name}...")
            
            # Compute per-class AP stats
            per_class_stats = compute_per_class_ap_stats(
                config["dir"], config["n_folds"], use_best_eval
            )
            
            if per_class_stats:
                model_entry["per_class_ap"] = per_class_stats
                print(f"    Added per-class AP for {len(per_class_stats)} tIoU thresholds")
            else:
                print(f"    No per-class AP data found")
    
    # Add/update single modality models
    existing_models = {m["model"]: i for i, m in enumerate(table["models"])}
    single_modality = ["V-JEPA", "PoseC3D", "STGCN++"]
    
    for model_name in single_modality:
        config = models_config[model_name]
        use_best_eval = config.get("use_best_eval", True)
        note = config.get("note", "")
        
        print(f"  {'Updating' if model_name in existing_models else 'Adding'} model: {model_name} ({note})...")
        
        # Compute mAP stats
        mAP_stats = compute_mAP_stats(config["dir"], config["n_folds"], use_best_eval)
        per_class_stats = compute_per_class_ap_stats(config["dir"], config["n_folds"], use_best_eval)
        
        if mAP_stats:
            new_entry = {
                "model": model_name,
                "folds": str(config["n_folds"]),
                "type": "single_modality",
                "hpo_status": note
            }
            
            # Add mAP metrics
            for key, val in mAP_stats.items():
                # Convert mAP@0.3 to mAP_0.3 format
                table_key = key.replace("@", "_")
                new_entry[table_key] = val
            
            if per_class_stats:
                new_entry["per_class_ap"] = per_class_stats
            
            # Update or add
            if model_name in existing_models:
                table["models"][existing_models[model_name]] = new_entry
            else:
                table["models"].append(new_entry)
            
            print(f"    Updated with mAP stats")
        else:
            print(f"    No mAP data found")
    
    # Save updated table
    with open(table_path, 'w') as f:
        json.dump(table, f, indent=2)
    
    print(f"Saved: {table_path}")


def update_table_5_9():
    """Update Table 5.9 with missing stdev for binary TAL."""
    print("\n=== Updating Table 5.9 (Binary TAL) ===")
    
    table_path = TABLES_DIR / "table_5_9_binary_tal.json"
    
    with open(table_path) as f:
        table = json.load(f)
    
    # Check V-JEPA Binary cv_summary for stdev
    vjepa_binary_summary = TAL_EVAL_DIR / "vjepa_binary" / "cv_summary.json"
    
    if vjepa_binary_summary.exists():
        with open(vjepa_binary_summary) as f:
            summary = json.load(f)
        
        # Update the "V-JEPA Binary (window)" entry
        for model_entry in table["models"]:
            if "V-JEPA Binary (window)" in model_entry.get("model", ""):
                print(f"  Updating {model_entry['model']}...")
                
                # Extract stdev from summary
                key_mapping = {
                    "mAP_0.3": "mAP@0.3_std",
                    "mAP_0.5": "mAP@0.5_std",
                    "mAP_0.7": "mAP@0.7_std",
                    "avg_mAP": "avg_mAP_std"
                }
                
                for key, std_key in key_mapping.items():
                    if std_key in summary and key in model_entry:
                        model_entry[key]["std"] = round(summary[std_key] * 100, 2)
                
                print(f"    Updated with stdev from cv_summary")
    
    # Save updated table
    with open(table_path, 'w') as f:
        json.dump(table, f, indent=2)
    
    print(f"Saved: {table_path}")


def create_comprehensive_tal_table():
    """Create a comprehensive TAL summary with all per-class AP data."""
    print("\n=== Creating Comprehensive TAL Summary ===")
    
    output_path = TABLES_DIR / "table_tal_comprehensive.json"
    
    models = {
        "ActionFormer + V-JEPA": {
            "dir": TAL_EVAL_DIR / "actionformer_vjepa",
            "n_folds": 2,
            "type": "end_to_end"
        },
        "V-JEPA + PoseC3D (MLP)": {
            "dir": TAL_EVAL_DIR / "vjepa_posec3d_mlp_logp",
            "n_folds": 2,
            "type": "fusion"
        },
        "V-JEPA + STGCN++ (MLP)": {
            "dir": TAL_EVAL_DIR / "vjepa_stgcnpp_mlp_logp",
            "n_folds": 2,
            "type": "fusion"
        },
        "V-JEPA Binary": {
            "dir": TAL_EVAL_DIR / "vjepa_binary",
            "n_folds": 3,
            "type": "window_binary"
        }
    }
    
    results = {
        "task": "temporal_action_localization",
        "models": []
    }
    
    for model_name, config in models.items():
        print(f"  Processing {model_name}...")
        
        model_result = {
            "model": model_name,
            "type": config["type"],
            "n_folds": config["n_folds"]
        }
        
        # Get mAP stats
        mAP_stats = compute_mAP_stats(config["dir"], config["n_folds"])
        if mAP_stats:
            model_result["mAP"] = mAP_stats
            print(f"    mAP stats: {list(mAP_stats.keys())}")
        
        # Get per-class AP stats
        per_class_stats = compute_per_class_ap_stats(config["dir"], config["n_folds"])
        if per_class_stats:
            model_result["per_class_ap"] = per_class_stats
            print(f"    Per-class AP: {len(per_class_stats)} tIoU thresholds")
        
        results["models"].append(model_result)
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved: {output_path}")


def main():
    print("Updating TAL Tables with Standard Deviations")
    print("=" * 60)
    
    update_table_5_7()
    update_table_5_9()
    create_comprehensive_tal_table()
    
    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
