#!/bin/bash
#
# Evaluate V-JEPA Binary (RMM vs BG) model for TAL
#
# This script:
# 1. Converts predictions_clip.csv to TAL format for each fold
# 2. Runs grid search for optimal postprocessing params
# 3. Evaluates with best params and generates CV summary
#
# Usage:
#   ./eval_vjepa_binary_cv.sh
#

set -e

# Configuration
MODEL_NAME="vjepa_binary"
MODEL_DISPLAY_NAME="V-JEPA Binary (RMM vs BG)"
NUM_FOLDS=3

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAL_DIR="$(dirname "${SCRIPT_DIR}")"
# Repo root = parent of tal/ (derived from this script's own location above).
REPO_ROOT="$(dirname "${TAL_DIR}")"
VJEPA_RUNS="${REPO_ROOT}/v-jepa/runs/vjepa2_tal_cv_binary_balanced"
OUT_DIR="${TAL_DIR}/eval_results/${MODEL_NAME}"
SPLITS_ROOT="${REPO_ROOT}/dataprep/splits"

echo "=============================================="
echo "V-JEPA Binary TAL Evaluation"
echo "=============================================="
echo "Model: ${MODEL_DISPLAY_NAME}"
echo "Folds: ${NUM_FOLDS}"
echo "Output: ${OUT_DIR}"
echo ""

cd "${TAL_DIR}"

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyskl

# ============================================================================
# Step 1: Convert predictions to TAL format for each fold
# ============================================================================
echo "----------------------------------------------"
echo "Step 1: Converting predictions to TAL format"
echo "----------------------------------------------"

for FOLD in $(seq 0 $((NUM_FOLDS - 1))); do
    INPUT_CSV="${VJEPA_RUNS}/fold_${FOLD}/predictions_clip.csv"
    FOLD_OUT="${OUT_DIR}/fold${FOLD}"
    TAL_CSV="${FOLD_OUT}/tal_format_preds.csv"
    
    if [ ! -f "${INPUT_CSV}" ]; then
        echo "Fold ${FOLD}: Skipping (predictions not found: ${INPUT_CSV})"
        continue
    fi
    
    mkdir -p "${FOLD_OUT}"
    
    if [ ! -f "${TAL_CSV}" ]; then
        echo "Fold ${FOLD}: Converting to TAL format..."
        python convert_preds_to_tal_format.py \
            -i "${INPUT_CSV}" \
            -o "${TAL_CSV}"
    else
        echo "Fold ${FOLD}: TAL format exists, skipping conversion"
    fi
done

echo ""

# ============================================================================
# Step 2 & 3: Grid search and evaluation with best params
# ============================================================================
echo "----------------------------------------------"
echo "Step 2-3: Grid search and evaluation"
echo "----------------------------------------------"

python << 'EOF'
import json
import sys
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(".").resolve()))
sys.path.insert(0, str(Path("scripts").resolve()))

import numpy as np
import pandas as pd
from eval_binary_tal import (
    load_binary_gt_segments,
    evaluate_binary_tal,
    save_binary_metrics,
    format_binary_report,
)

# Configuration
MODEL_NAME = "vjepa_binary"
MODEL_DISPLAY_NAME = "V-JEPA Binary (RMM vs BG)"
NUM_FOLDS = 3
TAL_DIR = Path.cwd()  # script cd's into tal/ before this heredoc
EVAL_RESULTS_DIR = TAL_DIR / "eval_results"
SPLITS_ROOT = TAL_DIR.parent / "dataprep" / "splits"

# Parameter grid
PARAM_GRID = {
    "thr": [0.3, 0.4, 0.5, 0.6],
    "smooth_k": [1, 3, 5, 7],
    "merge_gap_sec": [0.5, 1.0, 1.5, 2.0],
}
TIOU_THRESHOLDS = [0.3, 0.5, 0.7]

print("Loading GT segments (all RMM types merged)...")
gt_by_video = load_binary_gt_segments(SPLITS_ROOT, mode="cv")
n_gt = sum(len(v) for v in gt_by_video.values())
print(f"  Loaded {n_gt} GT segments from {len(gt_by_video)} videos")

# Collect available folds
fold_data = []
for fold_idx in range(NUM_FOLDS):
    csv_path = EVAL_RESULTS_DIR / MODEL_NAME / f"fold{fold_idx}" / "tal_format_preds.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        fold_data.append((fold_idx, csv_path, df))
        print(f"  Fold {fold_idx}: Loaded {len(df)} windows")
    else:
        print(f"  Fold {fold_idx}: CSV not found, skipping")

if not fold_data:
    print("ERROR: No fold CSVs found!")
    sys.exit(1)

print(f"\nRunning grid search over {len(fold_data)} folds...")

best_avg_map = -1
best_params = None
results = []

param_combos = list(product(
    PARAM_GRID["thr"],
    PARAM_GRID["smooth_k"],
    PARAM_GRID["merge_gap_sec"]
))

for thr, smooth_k, merge_gap_sec in param_combos:
    pp_params = {"thr": thr, "smooth_k": smooth_k, "merge_gap_sec": merge_gap_sec}
    
    fold_maps = []
    for fold_idx, csv_path, df in fold_data:
        metrics = evaluate_binary_tal(df, gt_by_video, pp_params, TIOU_THRESHOLDS)
        fold_maps.append(metrics.get("avg_mAP", 0.0) if not np.isnan(metrics.get("avg_mAP", 0.0)) else 0.0)
    
    avg_map = np.mean(fold_maps)
    results.append({
        "thr": thr,
        "smooth_k": smooth_k,
        "merge_gap_sec": merge_gap_sec,
        "avg_mAP": avg_map,
    })
    
    if avg_map > best_avg_map:
        best_avg_map = avg_map
        best_params = pp_params.copy()

print(f"\nBest params: {best_params}")
print(f"Best avg_mAP: {best_avg_map:.4f}")

# Save best params
best_params_file = EVAL_RESULTS_DIR / "best_params_per_model.json"
if best_params_file.exists():
    with open(best_params_file) as f:
        all_best_params = json.load(f)
else:
    all_best_params = {}

all_best_params[MODEL_NAME] = {
    "best_params": best_params,
    "best_avg_mAP": best_avg_map,
}

with open(best_params_file, "w") as f:
    json.dump(all_best_params, f, indent=2)

# Save grid search results
grid_results_df = pd.DataFrame(results)
model_out_dir = EVAL_RESULTS_DIR / MODEL_NAME
grid_results_df.to_csv(model_out_dir / "grid_search_results.csv", index=False)

print("\n" + "=" * 60)
print("Final Evaluation with Best Params")
print("=" * 60)

fold_metrics = {}

for fold_idx, csv_path, df in fold_data:
    print(f"\nFold {fold_idx}: Evaluating...")
    
    metrics = evaluate_binary_tal(df, gt_by_video, best_params, TIOU_THRESHOLDS)
    
    # Save fold artifacts
    fold_out_dir = model_out_dir / f"fold{fold_idx}" / "best_eval"
    save_binary_metrics(metrics, fold_out_dir)
    
    fold_metrics[fold_idx] = metrics
    
    print(f"  mAP@0.5: {metrics.get('mAP@0.5', 0):.4f}")
    print(f"  Recall@0.5: {metrics.get('Recall@0.5', 0):.4f}")
    print(f"  avg_mAP: {metrics.get('avg_mAP', 0):.4f}")
    print(f"  Saved to: {fold_out_dir}")

# Generate CV summary
print("\n" + "=" * 60)
print("CV SUMMARY")
print("=" * 60)

agg_keys = [
    "mAP@0.3", "mAP@0.5", "mAP@0.7", "avg_mAP",
    "Recall@0.3", "Recall@0.5", "Recall@0.7", "avg_Recall",
    "Precision@0.3", "Precision@0.5", "Precision@0.7",
]

cv_summary = {
    "model": MODEL_DISPLAY_NAME,
    "n_folds": len(fold_metrics),
    "folds_evaluated": sorted(fold_metrics.keys()),
    "best_params": best_params,
}

for key in agg_keys:
    values = [
        m[key] for m in fold_metrics.values()
        if key in m and m[key] is not None and not np.isnan(m[key])
    ]
    if values:
        cv_summary[f"{key}_mean"] = float(np.mean(values))
        cv_summary[f"{key}_std"] = float(np.std(values))
        cv_summary[f"{key}_values"] = values
    else:
        cv_summary[f"{key}_mean"] = None
        cv_summary[f"{key}_std"] = None

cv_summary["total_windows"] = sum(m.get("n_windows", 0) for m in fold_metrics.values())
cv_summary["total_segments"] = sum(m.get("n_segments", 0) for m in fold_metrics.values())

# Get aggregate n_gt from one of the metrics (should be same for all folds as we load all GT)
if fold_metrics:
    first_metrics = list(fold_metrics.values())[0]
    cv_summary["n_gt_total"] = first_metrics.get("details", {}).get("tIoU=0.5", {}).get("n_gt", 0)

# Save CV summary
cv_json_path = model_out_dir / "cv_summary.json"
with open(cv_json_path, "w") as f:
    json.dump(cv_summary, f, indent=2)

# Print summary
for key in ["mAP@0.3", "mAP@0.5", "mAP@0.7", "avg_mAP", "Recall@0.5", "avg_Recall"]:
    mean_val = cv_summary.get(f"{key}_mean")
    std_val = cv_summary.get(f"{key}_std", 0)
    if mean_val is not None:
        print(f"  {key}: {mean_val:.4f} ± {std_val:.4f}")
    else:
        print(f"  {key}: N/A")

print("=" * 60)
print(f"\nCV summary saved to: {cv_json_path}")
EOF

echo ""
echo "=============================================="
echo "Evaluation complete!"
echo "Results saved to: ${OUT_DIR}"
echo "=============================================="


