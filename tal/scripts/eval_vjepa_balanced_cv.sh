#!/bin/bash
#
# Evaluate V-JEPA Balanced (5-class) model for TAL (4-class evaluation)
#
# This script:
# 1. Converts predictions_clip.csv to TAL format for each fold
# 2. Runs grid search for optimal postprocessing params
# 3. Evaluates with best params and generates CV summary
#
# Usage:
#   ./eval_vjepa_balanced_cv.sh
#

set -e

# Configuration
MODEL_NAME="vjepa_balanced"
MODEL_DISPLAY_NAME="V-JEPA Balanced"
NUM_FOLDS=3

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAL_DIR="$(dirname "${SCRIPT_DIR}")"
# Repo root = parent of tal/ (derived from this script's own location above).
REPO_ROOT="$(dirname "${TAL_DIR}")"
VJEPA_RUNS="${REPO_ROOT}/v-jepa/runs/vjepa2_tal_cv_5class_balanced"
OUT_DIR="${TAL_DIR}/eval_results/${MODEL_NAME}"
SPLITS_ROOT="${REPO_ROOT}/dataprep/splits"

echo "=============================================="
echo "V-JEPA Balanced TAL Evaluation"
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
# Step 2: Run grid search for optimal postprocessing params
# ============================================================================
echo "----------------------------------------------"
echo "Step 2: Grid search for postprocessing params"
echo "----------------------------------------------"

# Add the model to grid search if not already there
python << 'EOF'
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))

from pathlib import Path
import numpy as np
import pandas as pd
from window_to_segments import PostprocessParams, window_scores_to_segments
from tal_map_eval import compute_map, load_gt_segments, ID2LABEL_4CLASS

# Configuration
MODEL_NAME = "vjepa_balanced"
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
CLASS_IDS = [0, 1, 2, 3]  # 4-class evaluation

print(f"Loading GT segments...")
gt_by_video = load_gt_segments(SPLITS_ROOT, "4class", "cv")
n_gt = sum(len(v) for v in gt_by_video.values())
print(f"  Loaded {n_gt} GT segments from {len(gt_by_video)} videos")

# Collect available folds
fold_csvs = []
for fold_idx in range(NUM_FOLDS):
    csv_path = EVAL_RESULTS_DIR / MODEL_NAME / f"fold{fold_idx}" / "tal_format_preds.csv"
    if csv_path.exists():
        fold_csvs.append((fold_idx, csv_path))
    else:
        print(f"  Fold {fold_idx}: CSV not found, skipping")

if not fold_csvs:
    print("ERROR: No fold CSVs found!")
    sys.exit(1)

print(f"\nRunning grid search over {len(fold_csvs)} folds...")

best_avg_map = -1
best_params = None
results = []

# Iterate over parameter combinations
from itertools import product
param_combos = list(product(
    PARAM_GRID["thr"],
    PARAM_GRID["smooth_k"],
    PARAM_GRID["merge_gap_sec"]
))

for thr, smooth_k, merge_gap_sec in param_combos:
    pp_params = PostprocessParams(
        smooth_k=smooth_k,
        threshold=thr,
        merge_gap_sec=merge_gap_sec,
        min_duration_sec=0.0,
        score_reducer="max",
        class_ids=CLASS_IDS,
    )
    
    fold_maps = []
    for fold_idx, csv_path in fold_csvs:
        df = pd.read_csv(csv_path)
        segments_df = window_scores_to_segments(df, pp_params)
        
        if segments_df.empty:
            fold_maps.append(0.0)
            continue
        
        metrics = compute_map(segments_df, gt_by_video, CLASS_IDS, TIOU_THRESHOLDS)
        fold_maps.append(metrics.get("avg_mAP", 0.0))
    
    avg_map = np.mean(fold_maps)
    results.append({
        "thr": thr,
        "smooth_k": smooth_k,
        "merge_gap_sec": merge_gap_sec,
        "avg_mAP": avg_map,
    })
    
    if avg_map > best_avg_map:
        best_avg_map = avg_map
        best_params = {"thr": thr, "smooth_k": smooth_k, "merge_gap_sec": merge_gap_sec}

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

print(f"Saved best params to: {best_params_file}")

# Save grid search results
grid_results_df = pd.DataFrame(results)
grid_results_file = EVAL_RESULTS_DIR / MODEL_NAME / "grid_search_results.csv"
grid_results_df.to_csv(grid_results_file, index=False)
print(f"Saved grid search results to: {grid_results_file}")
EOF

echo ""

# ============================================================================
# Step 3: Evaluate with best params and generate CV summary
# ============================================================================
echo "----------------------------------------------"
echo "Step 3: Final evaluation with best params"
echo "----------------------------------------------"

python << 'EOF'
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))

import numpy as np
import pandas as pd
from window_to_segments import PostprocessParams, window_scores_to_segments
from tal_map_eval import (
    compute_map, load_gt_segments, format_metrics_report, save_metrics,
    ID2LABEL_4CLASS,
)

# Configuration
MODEL_NAME = "vjepa_balanced"
MODEL_DISPLAY_NAME = "V-JEPA Balanced"
NUM_FOLDS = 3
TAL_DIR = Path.cwd()  # script cd's into tal/ before this heredoc
EVAL_RESULTS_DIR = TAL_DIR / "eval_results"
SPLITS_ROOT = TAL_DIR.parent / "dataprep" / "splits"

TIOU_THRESHOLDS = [0.3, 0.5, 0.7]
CLASS_IDS = [0, 1, 2, 3]

# Load best params
best_params_file = EVAL_RESULTS_DIR / "best_params_per_model.json"
with open(best_params_file) as f:
    all_best_params = json.load(f)

best_params = all_best_params[MODEL_NAME]["best_params"]
print(f"Using best params: {best_params}")

# Load GT
gt_by_video = load_gt_segments(SPLITS_ROOT, "4class", "cv")

# Evaluate each fold
fold_metrics = {}

for fold_idx in range(NUM_FOLDS):
    csv_path = EVAL_RESULTS_DIR / MODEL_NAME / f"fold{fold_idx}" / "tal_format_preds.csv"
    
    if not csv_path.exists():
        print(f"Fold {fold_idx}: Skipping (CSV not found)")
        continue
    
    print(f"\nFold {fold_idx}: Evaluating...")
    
    df = pd.read_csv(csv_path)
    
    pp_params = PostprocessParams(
        smooth_k=best_params["smooth_k"],
        threshold=best_params["thr"],
        merge_gap_sec=best_params["merge_gap_sec"],
        min_duration_sec=0.0,
        score_reducer="max",
        class_ids=CLASS_IDS,
    )
    
    segments_df = window_scores_to_segments(df, pp_params)
    metrics = compute_map(segments_df, gt_by_video, CLASS_IDS, TIOU_THRESHOLDS)
    
    # Add metadata
    metrics["n_windows"] = len(df)
    metrics["n_segments"] = len(segments_df)
    metrics["n_videos"] = df["video_key"].nunique()
    metrics["postprocess_params"] = best_params
    
    # Save fold artifacts
    fold_out_dir = EVAL_RESULTS_DIR / MODEL_NAME / f"fold{fold_idx}" / "best_eval"
    fold_out_dir.mkdir(parents=True, exist_ok=True)
    
    segments_df.to_csv(fold_out_dir / "pred_segments.csv", index=False)
    save_metrics(metrics, fold_out_dir, CLASS_IDS, ID2LABEL_4CLASS)
    
    # Save eval config
    config = {
        "pred_csv": str(csv_path),
        "postprocess": best_params,
        "class_ids": CLASS_IDS,
        "tiou_thresholds": TIOU_THRESHOLDS,
    }
    with open(fold_out_dir / "eval_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    fold_metrics[fold_idx] = metrics
    
    print(f"  mAP@0.5: {metrics.get('mAP@0.5', 0):.4f}")
    print(f"  Recall@0.5: {metrics.get('Recall@0.5', 0):.4f}")
    print(f"  avg_mAP: {metrics.get('avg_mAP', 0):.4f}")
    print(f"  Saved to: {fold_out_dir}")

# Generate CV summary
print("\n" + "=" * 60)
print("CV SUMMARY")
print("=" * 60)

agg_keys = ["mAP@0.3", "mAP@0.5", "mAP@0.7", "avg_mAP", "Recall@0.3", "Recall@0.5", "Recall@0.7", "avg_Recall"]

cv_summary = {
    "model": MODEL_DISPLAY_NAME,
    "n_folds": len(fold_metrics),
    "folds_evaluated": sorted(fold_metrics.keys()),
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

# Save CV summary
model_out_dir = EVAL_RESULTS_DIR / MODEL_NAME
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


