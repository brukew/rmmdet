#!/bin/bash
#SBATCH --job-name=posec3d_tal_bgsub
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/posec3d_tal_4class_cv_bgsub_%j.out
#SBATCH --error=slurm-logs/posec3d_tal_4class_cv_bgsub_%j.err

# ============================================================================
# PoseC3D Training: TAL 4-class CV with Background Subsampling
# 
# Trains on 5 classes (4 RMM + background) with class weighting and
# background subsampling (20%) to address class imbalance.
#
# Only processes fold 0 and 1 (fold 2 pickle pending).
# ============================================================================

# --- Portable repo-root resolution (auto-inserted) --------------------------
# Locate the repo root (the directory containing paths.py) so this script runs
# from any clone name/location, under `bash` or `sbatch`. SLURM copies the
# script to a spool dir, so if the script path does not resolve we fall back to
# $SLURM_SUBMIT_DIR then $PWD. Override by exporting REPO_ROOT before launch.
_rmm_find_root() {
  local d="$1"
  while [ -n "$d" ] && [ "$d" != "/" ]; do
    if [ -f "$d/paths.py" ]; then printf '%s\n' "$d"; return 0; fi
    d="$(dirname "$d")"
  done
  return 1
}
if [ -z "${REPO_ROOT:-}" ] || [ ! -f "${REPO_ROOT:-x}/paths.py" ]; then
  REPO_ROOT="$(_rmm_find_root "$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")" 2>/dev/null && pwd)")" \
    || REPO_ROOT="$(_rmm_find_root "${SLURM_SUBMIT_DIR:-$PWD}")" \
    || REPO_ROOT="$(_rmm_find_root "$PWD")" || true
fi
if [ -z "${REPO_ROOT:-}" ] || [ ! -f "${REPO_ROOT}/paths.py" ]; then
  echo "ERROR: cannot locate repo root (paths.py). cd to the repo or export REPO_ROOT." >&2
  exit 1
fi
export REPO_ROOT
# --- end repo-root resolution ----------------------------------------------

set -eo pipefail

echo "=========================================="
echo "PoseC3D TAL Training: 4-class CV with Background Subsampling"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

# Ensure logs directory exists
mkdir -p ${REPO_ROOT}/pyskl/logs

# Ensure real-time logging
export PYTHONUNBUFFERED=1

if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

conda activate pyskl

cd ${REPO_ROOT}/pyskl

# Configuration
CONFIG="configs/posec3d/slowonly_r50_sails_k400p/joint_tal_5class.py"
BASE_ANN_DIR="data/sails/tal/cv_4class/5class_windows_conf04"
BASE_WORK_DIR="work_dirs/posec3d/tal/cv_4class_5class_bgsub"
EPOCHS=20
LR=0.00125
BG_SUBSAMPLE=0.2

echo ""
echo "Config: $CONFIG"
echo "Annotation dir: $BASE_ANN_DIR"
echo "Work dir: $BASE_WORK_DIR"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LR"
echo "Background subsample: $BG_SUBSAMPLE"
echo ""

# Process fold 0 and 1 only (fold 2 pending)
for FOLD in 0 1; do
    ANN_FILE="${BASE_ANN_DIR}/fold${FOLD}.pkl"
    WORK_DIR="${BASE_WORK_DIR}/fold${FOLD}"
    
    echo ""
    echo "######################################################"
    echo "# FOLD ${FOLD}"
    echo "######################################################"
    echo "Annotation: $ANN_FILE"
    echo "Work dir: $WORK_DIR"
    
    # Check if pickle exists
    if [ ! -f "$ANN_FILE" ]; then
        echo "ERROR: Annotation file not found: $ANN_FILE"
        continue
    fi
    
    # Train with class weights and background subsampling
    python tools/train_weighted.py $CONFIG \
        --ann-file $ANN_FILE \
        --work-dir $WORK_DIR \
        --total-epochs $EPOCHS \
        --lr $LR \
        --bg-subsample $BG_SUBSAMPLE \
        --validate \
        --launcher none
    
    # Evaluate on val set
    BEST_CKPT=$(ls -t ${WORK_DIR}/best_*.pth 2>/dev/null | head -1)
    if [ -z "$BEST_CKPT" ]; then
        BEST_CKPT="${WORK_DIR}/latest.pth"
    fi
    
    echo ""
    echo "Evaluating fold ${FOLD}: $BEST_CKPT"
    
    python tools/evaluate_sails.py $CONFIG \
        -C $BEST_CKPT \
        --split val \
        --output-dir ${WORK_DIR}/eval_val \
        --cfg-options \
            data.train.dataset.ann_file=$ANN_FILE \
            data.val.ann_file=$ANN_FILE \
            model.cls_head.num_classes=5
done

# CV Summary (partial - fold 0 and 1 only)
echo ""
echo "=========================================="
echo "Computing Partial CV Summary (Folds 0-1)"
echo "=========================================="

python -c "
import json
import numpy as np
from pathlib import Path

base_dir = Path('${BASE_WORK_DIR}')
folds = []

for fold in range(2):  # Only 0 and 1
    metrics_file = base_dir / f'fold{fold}' / 'eval_val' / 'metrics.json'
    if metrics_file.exists():
        with open(metrics_file) as f:
            m = json.load(f)
            m['fold'] = fold
            folds.append(m)
            print(f'Fold {fold}: loaded metrics')
    else:
        print(f'Fold {fold}: metrics not found')

if folds:
    summary = {'per_fold': folds, 'note': 'Partial CV (fold 0-1 only, fold 2 pending)'}
    metric_keys = [k for k in folds[0].keys() if k != 'fold']
    for key in metric_keys:
        vals = [f[key] for f in folds if key in f]
        if vals and isinstance(vals[0], (int, float)):
            summary[f'{key}_mean'] = float(np.mean(vals))
            summary[f'{key}_std'] = float(np.std(vals))
    
    with open(base_dir / 'cv_summary_partial.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print()
    print('Partial CV Summary (Folds 0-1, with BG Subsampling):')
    print(f'  Clip Top-1: {summary.get(\"clip_top1_acc_mean\", summary.get(\"top1_acc_mean\", 0)):.2f} ± {summary.get(\"clip_top1_acc_std\", summary.get(\"top1_acc_std\", 0)):.2f}%')
    print(f'  Clip Macro-F1: {summary.get(\"clip_macro_f1_mean\", 0):.2f} ± {summary.get(\"clip_macro_f1_std\", 0):.2f}%')
else:
    print('No fold metrics found yet.')
"

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="


