#!/bin/bash
#
# SAILS PoseC3D Cross-Validation Training Script
#
# Runs 3-fold CV training sequentially and aggregates results.
#
# Usage:
#   bash scripts/train_sails_cv.sh [OPTIONS]
#
# Options:
#   --conf CONF         Keypoint confidence threshold (default: 0.6)
#   --epochs EPOCHS     Number of training epochs (default: 12)
#   --lr LR             Learning rate (default: 0.01)
#   --data-dir DIR      Directory containing CV pickles (default: data/sails/cv)
#   --work-dir DIR      Base work directory (default: work_dirs/posec3d/cv)
#   --config PATH       Config file (default: configs/posec3d/slowonly_r50_sails_k400p/joint.py)
#   --folds "0 1 2"     Which folds to run (default: "0 1 2")
#   --dry-run           Print commands without executing
#
# Example:
#   bash scripts/train_sails_cv.sh --conf 0.6 --epochs 12
#   bash scripts/train_sails_cv.sh --folds "0 1" --dry-run

set -e

# ============================================================================
# DEFAULT PARAMETERS
# ============================================================================

CONF="0.6"
EPOCHS="12"
LR="0.01"
DATA_DIR="data/sails/cv"
WORK_DIR="work_dirs/posec3d/cv"
CONFIG="configs/posec3d/slowonly_r50_sails_k400p/joint.py"
FOLDS="0 1 2"
DRY_RUN=false
LAUNCHER="none"

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --conf)
            CONF="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --work-dir)
            WORK_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --folds)
            FOLDS="$2"
            shift 2
            ;;
        --launcher)
            LAUNCHER="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            head -30 "$0" | tail -25
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# CONFIGURATION
# ============================================================================

# Format confidence for directory name (0.6 -> conf06)
CONF_DIR="conf$(echo $CONF | tr -d '.')"

# Pickle directory
PICKLE_DIR="${DATA_DIR}/${CONF_DIR}"

# Work directory for this run
RUN_WORK_DIR="${WORK_DIR}/${CONF_DIR}"

echo "============================================================"
echo "SAILS PoseC3D Cross-Validation Training"
echo "============================================================"
echo "Configuration:"
echo "  Confidence threshold: ${CONF}"
echo "  Epochs: ${EPOCHS}"
echo "  Learning rate: ${LR}"
echo "  Pickle directory: ${PICKLE_DIR}"
echo "  Work directory: ${RUN_WORK_DIR}"
echo "  Config: ${CONFIG}"
echo "  Folds: ${FOLDS}"
echo "  Launcher: ${LAUNCHER}"
echo ""

# Check if pickle directory exists
if [ ! -d "$PICKLE_DIR" ]; then
    echo "ERROR: Pickle directory not found: ${PICKLE_DIR}"
    echo ""
    echo "Generate CV pickles first:"
    echo "  python tools/data/create_sails_annotations.py \\"
    echo "      --mode cv \\"
    echo "      --splits-dir /path/to/cv_splits_4class/ \\"
    echo "      --output-dir ${DATA_DIR} \\"
    echo "      --min-keypoint-conf ${CONF}"
    exit 1
fi

# ============================================================================
# TRAINING LOOP
# ============================================================================

RESULTS_FILE="${RUN_WORK_DIR}/cv_results.json"
mkdir -p "${RUN_WORK_DIR}"

# Initialize results tracking
echo "{" > "${RESULTS_FILE}"
echo '  "folds": {' >> "${RESULTS_FILE}"

FOLD_COUNT=0
TOTAL_FOLDS=$(echo $FOLDS | wc -w)

for FOLD in $FOLDS; do
    FOLD_COUNT=$((FOLD_COUNT + 1))
    
    echo ""
    echo "============================================================"
    echo "FOLD ${FOLD} (${FOLD_COUNT}/${TOTAL_FOLDS})"
    echo "============================================================"
    
    FOLD_PICKLE="${PICKLE_DIR}/fold${FOLD}.pkl"
    FOLD_WORK_DIR="${RUN_WORK_DIR}/fold${FOLD}"
    
    if [ ! -f "$FOLD_PICKLE" ]; then
        echo "ERROR: Pickle not found: ${FOLD_PICKLE}"
        continue
    fi
    
    echo "Pickle: ${FOLD_PICKLE}"
    echo "Output: ${FOLD_WORK_DIR}"
    echo ""
    
    # Build training command
    TRAIN_CMD="python tools/train.py ${CONFIG} \
        --work-dir ${FOLD_WORK_DIR} \
        --cfg-options ann_file=${FOLD_PICKLE} total_epochs=${EPOCHS} optimizer.lr=${LR} \
        --validate \
        --launcher ${LAUNCHER}"
    
    if $DRY_RUN; then
        echo "[DRY RUN] Would execute:"
        echo "  ${TRAIN_CMD}"
    else
        echo "Starting training..."
        eval $TRAIN_CMD
        
        # Find best checkpoint
        BEST_CKPT=$(ls -t ${FOLD_WORK_DIR}/best_*.pth 2>/dev/null | head -1)
        if [ -z "$BEST_CKPT" ]; then
            BEST_CKPT="${FOLD_WORK_DIR}/latest.pth"
        fi
        
        echo ""
        echo "Running evaluation on validation set..."
        
        EVAL_CMD="python tools/evaluate_sails.py ${CONFIG} \
            -C ${BEST_CKPT} \
            --split val \
            --output-dir ${FOLD_WORK_DIR}/eval_val \
            --cfg-options ann_file=${FOLD_PICKLE}"
        
        eval $EVAL_CMD
        
        # Extract metrics for aggregation
        if [ -f "${FOLD_WORK_DIR}/eval_val/metrics.json" ]; then
            echo "  \"fold${FOLD}\": $(cat ${FOLD_WORK_DIR}/eval_val/metrics.json)," >> "${RESULTS_FILE}"
        fi
    fi
done

# ============================================================================
# AGGREGATE RESULTS
# ============================================================================

if ! $DRY_RUN; then
    echo ""
    echo "============================================================"
    echo "AGGREGATING RESULTS"
    echo "============================================================"
    
    # Close the folds object and compute averages
    # Remove trailing comma and close JSON
    sed -i '$ s/,$//' "${RESULTS_FILE}"
    echo '  },' >> "${RESULTS_FILE}"
    
    # Compute averages using Python
    python3 << EOF
import json
import os
from pathlib import Path

work_dir = "${RUN_WORK_DIR}"
folds = "${FOLDS}".split()

# Collect metrics from all folds
all_metrics = []
for fold in folds:
    metrics_path = Path(work_dir) / f"fold{fold}" / "eval_val" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            all_metrics.append(json.load(f))

if not all_metrics:
    print("No metrics found to aggregate")
    exit(0)

# Compute averages
avg_metrics = {}
std_metrics = {}
metric_keys = [k for k in all_metrics[0].keys() if isinstance(all_metrics[0][k], (int, float))]

import numpy as np

for key in metric_keys:
    values = [m[key] for m in all_metrics if key in m]
    if values:
        avg_metrics[key] = float(np.mean(values))
        std_metrics[key] = float(np.std(values))

# Print summary
print("\n" + "=" * 60)
print("CROSS-VALIDATION RESULTS SUMMARY")
print("=" * 60)
print(f"Folds evaluated: {len(all_metrics)}")
print()

print("Clip-Level Metrics (mean ± std):")
for key in ['clip_top1_acc', 'clip_top2_acc', 'clip_macro_f1', 'clip_weighted_f1', 'clip_cohens_kappa']:
    if key in avg_metrics:
        if 'kappa' in key:
            print(f"  {key}: {avg_metrics[key]:.4f} ± {std_metrics[key]:.4f}")
        else:
            print(f"  {key}: {avg_metrics[key]:.2f}% ± {std_metrics[key]:.2f}%")

print()
print("Video-Level Metrics (mean ± std):")
for key in ['video_top1_acc', 'video_top2_acc', 'video_macro_f1', 'video_weighted_f1', 'video_cohens_kappa']:
    if key in avg_metrics:
        if 'kappa' in key:
            print(f"  {key}: {avg_metrics[key]:.4f} ± {std_metrics[key]:.4f}")
        else:
            print(f"  {key}: {avg_metrics[key]:.2f}% ± {std_metrics[key]:.2f}%")

# Save aggregated results
summary = {
    "num_folds": len(all_metrics),
    "folds": folds,
    "average": avg_metrics,
    "std": std_metrics,
}

summary_path = Path(work_dir) / "cv_summary.json"
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary saved to: {summary_path}")
EOF

fi

echo ""
echo "============================================================"
echo "CV Training Complete!"
echo "============================================================"
echo "Results directory: ${RUN_WORK_DIR}"
echo ""







