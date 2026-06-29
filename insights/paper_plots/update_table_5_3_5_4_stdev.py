#!/usr/bin/env python3
"""
Update Table 5.3 and 5.4 with standard deviations for per-class precision and recall.

This script computes per-class precision/recall per fold from the model predictions
and adds stdev to the JSON files.

Usage:
    python update_table_5_3_5_4_stdev.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.metrics import confusion_matrix

# Project root
ACTREG_ROOT = Path(__file__).parent.parent.parent

# Paths
TABLES_DIR = ACTREG_ROOT / "insights" / "tables"

# Model paths for 4-class task
MODELS_4CLASS = {
    "Fusion": {
        "base_dir": ACTREG_ROOT / "fusion" / "runs" / "4class_cv_3way",
        "fold_pattern": "fold_{fold}",
        "data_type": "predictions",  # predictions_clip.csv
    },
    "V-JEPA": {
        "base_dir": ACTREG_ROOT / "v-jepa" / "runs" / "vjepa2_rmm_cv" / "f64_lr1e-5_bs1_acc8_ep20_crop_4cls",
        "fold_pattern": "fold_{fold}",
        "data_type": "per_class_csv",  # per_class_clip.csv
    },
    "PoseC3D": {
        "base_dir": ACTREG_ROOT / "pyskl" / "work_dirs" / "posec3d" / "cv" / "4class_conf04",
        "fold_pattern": "fold{fold}",
        "data_type": "metrics_json",  # eval_val/metrics.json
    },
    "STGCN++": {
        "base_dir": ACTREG_ROOT / "pyskl" / "work_dirs" / "stgcnpp" / "cv" / "4class_conf04_4stream",
        "fold_pattern": "fold{fold}",
        "data_type": "predictions_direct",  # predictions_clip.csv in fold dir
    },
}

# Model paths for 5-class task
MODELS_5CLASS = {
    "Fusion": {
        "base_dir": ACTREG_ROOT / "fusion" / "runs" / "5class_cv_3way",
        "fold_pattern": "fold_{fold}",
        "data_type": "predictions",
    },
    "V-JEPA": {
        "base_dir": ACTREG_ROOT / "v-jepa" / "runs" / "vjepa2_rmm_type_cv",
        "fold_pattern": "fold_{fold}",
        "data_type": "per_class_csv",
    },
    "PoseC3D": {
        "base_dir": ACTREG_ROOT / "pyskl" / "work_dirs" / "posec3d" / "cv" / "5class_conf04_weighted",
        "fold_pattern": "fold{fold}",
        "data_type": "metrics_json",
    },
    "STGCN++": {
        "base_dir": ACTREG_ROOT / "pyskl" / "work_dirs" / "stgcnpp" / "cv" / "5class_conf04_4stream",
        "fold_pattern": "fold{fold}",
        "data_type": "predictions_direct",  # predictions_clip.csv in fold dir
    },
}

# Class names
CLASSES_4CLASS = ["hands flapping", "jumping", "rocking", "spinning"]
CLASSES_5CLASS = ["hands flapping", "jumping", "one hand flap", "rocking", "spinning"]


def load_per_class_csv(csv_path: Path) -> Dict[str, Dict[str, float]]:
    """
    Load per-class metrics from a CSV file (V-JEPA format).
    
    CSV format: class,precision,recall,f1,support
    """
    df = pd.read_csv(csv_path)
    
    results = {}
    for _, row in df.iterrows():
        class_name = row['class']
        results[class_name] = {
            'precision': row['precision'] * 100,  # Convert to percentage
            'recall': row['recall'] * 100
        }
    
    return results


def load_metrics_json(json_path: Path, class_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Load per-class metrics from a metrics.json file (pyskl format).
    
    JSON has keys like: clip_prec_hands flapping, clip_rec_hands flapping
    """
    with open(json_path) as f:
        data = json.load(f)
    
    results = {}
    for class_name in class_names:
        prec_key = f"clip_prec_{class_name}"
        rec_key = f"clip_rec_{class_name}"
        
        if prec_key in data and rec_key in data:
            results[class_name] = {
                'precision': data[prec_key],  # Already in percentage
                'recall': data[rec_key]
            }
    
    return results


def compute_pr_from_predictions(preds_csv: Path, class_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class precision and recall from predictions CSV.
    
    Handles both formats:
    - Fusion: segment_id, label_id, pred_id, ...
    - pyskl: segment_id, true_label, pred_label, true_class, pred_class, ...
    """
    df = pd.read_csv(preds_csv)
    n_classes = len(class_names)
    
    # Handle different column naming conventions
    if 'label_id' in df.columns:
        y_true = df['label_id'].values
        y_pred = df['pred_id'].values
    elif 'true_label' in df.columns:
        y_true = df['true_label'].values
        y_pred = df['pred_label'].values
    else:
        raise KeyError(f"Unknown column format in {preds_csv}")
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    
    results = {}
    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
        
        results[class_name] = {
            'precision': precision,
            'recall': recall
        }
    
    return results


def get_fold_metrics(
    model_config: Dict,
    fold: int,
    class_names: List[str]
) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Get per-class metrics for a single fold based on model configuration.
    """
    base_dir = model_config["base_dir"]
    fold_dir = base_dir / model_config["fold_pattern"].format(fold=fold)
    data_type = model_config["data_type"]
    
    if not fold_dir.exists():
        print(f"    Fold dir not found: {fold_dir}")
        return None
    
    try:
        if data_type == "predictions":
            preds_file = fold_dir / "predictions_clip.csv"
            if preds_file.exists():
                return compute_pr_from_predictions(preds_file, class_names)
        
        elif data_type == "per_class_csv":
            csv_file = fold_dir / "per_class_clip.csv"
            if csv_file.exists():
                return load_per_class_csv(csv_file)
        
        elif data_type == "metrics_json":
            json_file = fold_dir / "eval_val" / "metrics.json"
            if json_file.exists():
                return load_metrics_json(json_file, class_names)
        
        elif data_type == "metrics_json_direct":
            json_file = fold_dir / "metrics.json"
            if json_file.exists():
                return load_metrics_json(json_file, class_names)
        
        elif data_type == "predictions_direct":
            preds_file = fold_dir / "predictions_clip.csv"
            if preds_file.exists():
                return compute_pr_from_predictions(preds_file, class_names)
    
    except Exception as e:
        print(f"    Error loading fold {fold}: {e}")
        return None
    
    print(f"    Data file not found for fold {fold}")
    return None


def aggregate_pr_with_std(
    fold_results: List[Dict[str, Dict[str, float]]],
    class_names: List[str]
) -> Dict[str, Dict]:
    """
    Aggregate per-class P/R across folds with mean and std.
    """
    if not fold_results:
        return {}
    
    results = {}
    for class_name in class_names:
        precisions = []
        recalls = []
        
        for fr in fold_results:
            if class_name in fr:
                precisions.append(fr[class_name]["precision"])
                recalls.append(fr[class_name]["recall"])
        
        if precisions:
            results[class_name] = {
                "precision": {
                    "mean": round(float(np.mean(precisions)), 1),
                    "std": round(float(np.std(precisions)), 1)
                },
                "recall": {
                    "mean": round(float(np.mean(recalls)), 1),
                    "std": round(float(np.std(recalls)), 1)
                }
            }
        else:
            results[class_name] = {
                "precision": {"mean": None, "std": None},
                "recall": {"mean": None, "std": None}
            }
    
    return results


def process_model(
    model_name: str,
    model_config: Dict,
    class_names: List[str]
) -> Dict[str, Dict]:
    """
    Process a single model and return aggregated metrics.
    """
    print(f"  Processing {model_name}...")
    
    fold_results = []
    for fold in range(3):
        metrics = get_fold_metrics(model_config, fold, class_names)
        if metrics:
            fold_results.append(metrics)
    
    if fold_results:
        print(f"    Found {len(fold_results)} folds with data")
        return aggregate_pr_with_std(fold_results, class_names)
    else:
        print(f"    No per-fold data found")
        return {}


def update_table(
    table_path: Path,
    models_config: Dict,
    class_names: List[str],
    table_name: str
):
    """
    Update a table JSON file with stdev values.
    """
    print(f"\n=== Updating {table_name} ===")
    
    # Load existing table
    with open(table_path) as f:
        table = json.load(f)
    
    # Process each model
    model_results = {}
    for model_name, config in models_config.items():
        if model_name in table:
            result = process_model(model_name, config, class_names)
            if result:
                model_results[model_name] = result
    
    # VLM is zero-shot, no stdev available
    if "VLM" in table:
        print("  Processing VLM (zero-shot, no stdev)...")
        vlm_results = {}
        for class_name in class_names:
            if class_name in table["VLM"]["per_class"]:
                orig = table["VLM"]["per_class"][class_name]
                # Handle both old format (single value) and new format (dict)
                if isinstance(orig.get("precision"), dict):
                    prec_val = orig["precision"]["mean"]
                    rec_val = orig["recall"]["mean"]
                else:
                    prec_val = orig["precision"]
                    rec_val = orig["recall"]
                
                vlm_results[class_name] = {
                    "precision": {"mean": prec_val, "std": None},
                    "recall": {"mean": rec_val, "std": None}
                }
        model_results["VLM"] = vlm_results
    
    # Update table with new format
    new_table = {}
    for model_family, model_data in table.items():
        new_table[model_family] = {
            "model_name": model_data["model_name"],
            "per_class": {}
        }
        
        if model_family in model_results:
            new_table[model_family]["per_class"] = model_results[model_family]
        else:
            # Keep original format but convert to new structure if needed
            for class_name, metrics in model_data["per_class"].items():
                if isinstance(metrics.get("precision"), dict):
                    # Already in new format
                    new_table[model_family]["per_class"][class_name] = metrics
                else:
                    # Convert old format
                    new_table[model_family]["per_class"][class_name] = {
                        "precision": {"mean": metrics["precision"], "std": None},
                        "recall": {"mean": metrics["recall"], "std": None}
                    }
    
    # Save updated table
    with open(table_path, 'w') as f:
        json.dump(new_table, f, indent=2)
    
    print(f"Saved: {table_path}")


def main():
    print("Updating Tables 5.3 and 5.4 with standard deviations")
    print("=" * 60)
    
    # Update Table 5.3 (4-class)
    update_table(
        TABLES_DIR / "table_5_3_4class_pr.json",
        MODELS_4CLASS,
        CLASSES_4CLASS,
        "Table 5.3 (4-class)"
    )
    
    # Update Table 5.4 (5-class)
    update_table(
        TABLES_DIR / "table_5_4_5class_pr.json",
        MODELS_5CLASS,
        CLASSES_5CLASS,
        "Table 5.4 (5-class)"
    )
    
    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
