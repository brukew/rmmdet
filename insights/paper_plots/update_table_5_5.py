#!/usr/bin/env python3
"""
Update table_5_5_window_detection.json with missing metrics.

Fills in missing top2_acc, macro_f1, and cohens_kappa by:
1. Reading computed metrics from table_5_6_macro_f1_gain.json
2. Computing Cohen's kappa from raw predictions where available

Usage:
    python update_table_5_5.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

# Project root
ACTREG_ROOT = Path(__file__).parent.parent.parent

# Paths
EVAL_RESULTS_DIR = ACTREG_ROOT / "tal" / "eval_results"
DATAPREP_DIR = ACTREG_ROOT / "dataprep" / "tal"
TABLES_DIR = ACTREG_ROOT / "insights" / "tables"

# Map model names between table_5_5 and table_5_6
MODEL_NAME_MAP = {
    "V-JEPA": "vjepa",
    "V-JEPA (Balanced)": "vjepa_balanced",
    "V-JEPA Binary (Balanced)": None,  # Skip - binary model
    "PoseC3D (CE)": "posec3d_ce",
    "PoseC3D (Focal)": "posec3d_focal",
    "PoseC3D (CE Balanced)": "posec3d_ce_balanced",
    "STGCN++ 4-Stream (CE)": "stgcnpp_ce_4stream",
    "STGCN++ 4-Stream (CE Balanced)": None,  # No predictions available
    "STGCN++ 4-Stream (Focal)": "stgcnpp_focal_4stream",
    "V-JEPA + PoseC3D (MLP)": "vjepa_posec3d_mlp_logp",
    "V-JEPA + STGCN++ (MLP)": "vjepa_stgcnpp_mlp_logp",
    "V-JEPA + PoseC3D Bal (MLP)": "vjepa_posec3d_bal_mlp_logp",
}

GT_TO_PRED_CLASS = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    -1: 4
}


def load_predictions(model_dir: str, fold: int) -> pd.DataFrame:
    """Load window-level predictions."""
    pred_file = EVAL_RESULTS_DIR / model_dir / f"fold{fold}" / "tal_format_preds.csv"
    if not pred_file.exists():
        return None
    return pd.read_csv(pred_file)


def load_ground_truth(fold: int) -> pd.DataFrame:
    """Load ground truth window labels."""
    gt_file = DATAPREP_DIR / "splits_cv_4class" / f"fold_{fold}_val_windows.csv"
    if not gt_file.exists():
        return None
    df = pd.read_csv(gt_file)
    df["labels"] = df["labels"].apply(json.loads)
    return df


def compute_cohens_kappa(model_dir: str) -> dict:
    """
    Compute Cohen's kappa across CV folds.
    
    Returns:
        dict with mean and std, or None if no data
    """
    kappas = []
    
    for fold in range(3):
        preds_df = load_predictions(model_dir, fold)
        gt_df = load_ground_truth(fold)
        
        if preds_df is None or gt_df is None:
            continue
        
        merged = gt_df.merge(preds_df, on="window_id", how="inner")
        if len(merged) == 0:
            continue
        
        # Get predictions and ground truth
        score_cols = [f"score_class{i}" for i in range(5)]
        
        # Check if all score columns exist (not binary model)
        if not all(col in merged.columns for col in score_cols):
            continue
            
        scores = merged[score_cols].values
        y_pred = np.argmax(scores, axis=1)
        y_true = merged["primary_label"].map(GT_TO_PRED_CLASS).values
        
        kappa = cohen_kappa_score(y_true, y_pred)
        kappas.append(kappa)
    
    if len(kappas) == 0:
        return None
    
    return {
        "mean": round(float(np.mean(kappas)), 2),
        "std": round(float(np.std(kappas)), 2)
    }


def main():
    # Load existing table_5_5
    table_5_5_path = TABLES_DIR / "table_5_5_window_detection.json"
    with open(table_5_5_path) as f:
        table_5_5 = json.load(f)
    
    # Load table_5_6 for computed metrics
    table_5_6_path = TABLES_DIR / "table_5_6_macro_f1_gain.json"
    with open(table_5_6_path) as f:
        table_5_6 = json.load(f)
    
    # Create lookup from table_5_6 all_models
    computed_metrics = {}
    for model_dir, data in table_5_6.get("all_models", {}).items():
        computed_metrics[model_dir] = data
    
    print("Updating table_5_5 with missing metrics...\n")
    
    # Update each model in table_5_5
    for model_entry in table_5_5["models"]:
        model_name = model_entry["model"]
        model_dir = MODEL_NAME_MAP.get(model_name)
        
        print(f"Processing: {model_name}")
        
        # Skip if no mapping
        if model_dir is None:
            print(f"  Skipping - no predictions available")
            continue
        
        # Check if we have computed metrics
        if model_dir in computed_metrics:
            computed = computed_metrics[model_dir]
            
            # Update top2_acc if missing
            if model_entry.get("top2_acc") is None:
                if computed.get("top2_acc"):
                    model_entry["top2_acc"] = {
                        "mean": round(computed["top2_acc"]["mean"] * 100, 2),
                        "std": round(computed["top2_acc"]["std"] * 100, 2)
                    }
                    print(f"  Added top2_acc: {model_entry['top2_acc']}")
            
            # Update macro_f1 if missing
            if model_entry.get("macro_f1") is None:
                if computed.get("top1_f1"):
                    model_entry["macro_f1"] = {
                        "mean": round(computed["top1_f1"]["mean"] * 100, 2),
                        "std": round(computed["top1_f1"]["std"] * 100, 2)
                    }
                    print(f"  Added macro_f1: {model_entry['macro_f1']}")
        
        # Compute Cohen's kappa if missing
        if model_entry.get("cohens_kappa") is None:
            kappa = compute_cohens_kappa(model_dir)
            if kappa:
                model_entry["cohens_kappa"] = kappa
                print(f"  Computed cohens_kappa: {kappa}")
            else:
                print(f"  Could not compute cohens_kappa")
    
    # Save updated table_5_5
    with open(table_5_5_path, 'w') as f:
        json.dump(table_5_5, f, indent=2)
    
    print(f"\nSaved updated: {table_5_5_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("Summary of updates:")
    print("="*60)
    
    for model_entry in table_5_5["models"]:
        model_name = model_entry["model"]
        top2 = "✓" if model_entry.get("top2_acc") else "✗"
        f1 = "✓" if model_entry.get("macro_f1") else "✗"
        kappa = "✓" if model_entry.get("cohens_kappa") else "✗"
        print(f"  {model_name:<30} | top2: {top2} | f1: {f1} | κ: {kappa}")


if __name__ == "__main__":
    main()
