#!/bin/bash
#SBATCH --job-name=posec3d_tal_focal
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/orcd/data/satra/001/users/brukew/pyskl_logs/posec3d/tal/posec3d_tal_5class_cv_focal_%j.out
#SBATCH --error=/orcd/data/satra/001/users/brukew/pyskl_logs/posec3d/tal/posec3d_tal_5class_cv_focal_%j.err

# ============================================================================
# PoseC3D TAL Training: 5-class CV with Focal Loss + Early Stopping
# 
# Features:
#   - Focal loss (gamma=2) for hard example mining
#   - Inverse-frequency class weights (computed automatically)
#   - Background subsampling (20%) to reduce class imbalance
#   - Early stopping (patience=3) to prevent overfitting
#
# Runs on all 3 folds.
# ============================================================================

set -eo pipefail

echo "=========================================="
echo "PoseC3D TAL Training: 5-class CV with Focal Loss"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

# Ensure logs directory exists
mkdir -p /orcd/data/satra/001/users/brukew/pyskl_logs/posec3d/tal

# Ensure real-time logging
export PYTHONUNBUFFERED=1

if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyskl

cd /orcd/data/satra/001/users/brukew/actreg/pyskl

# Configuration
CONFIG="configs/posec3d/slowonly_r50_sails_k400p/joint_tal_5class_focal.py"
BASE_ANN_DIR="data/sails/tal/cv_4class/5class_windows_conf04"
BASE_WORK_DIR="work_dirs/posec3d/tal/cv_4class_5class_focal"
EPOCHS=24
LR=0.00125
BG_SUBSAMPLE=0.2

echo ""
echo "Config: $CONFIG"
echo "Annotation dir: $BASE_ANN_DIR"
echo "Work dir: $BASE_WORK_DIR"
echo "Epochs: $EPOCHS (with early stopping patience=3)"
echo "Learning rate: $LR"
echo "Background subsample: $BG_SUBSAMPLE"
echo "Loss: Focal Loss (gamma=2) + inverse class weights"
echo ""

# Process all 3 folds
for FOLD in 0 1 2; do
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
    
    # Train with focal loss, class weights, and background subsampling
    # train_weighted.py computes inverse-frequency weights and injects them
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

# CV Summary
echo ""
echo "=========================================="
echo "Computing CV Summary (All 3 Folds)"
echo "=========================================="

python -c "
import json
import numpy as np
from pathlib import Path

base_dir = Path('${BASE_WORK_DIR}')
folds = []

for fold in range(3):
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
    summary = {'per_fold': folds, 'model': 'PoseC3D', 'loss': 'focal', 'bg_subsample': 0.2}
    metric_keys = [k for k in folds[0].keys() if k != 'fold']
    for key in metric_keys:
        vals = [f[key] for f in folds if key in f]
        if vals and isinstance(vals[0], (int, float)):
            summary[f'{key}_mean'] = float(np.mean(vals))
            summary[f'{key}_std'] = float(np.std(vals))
    
    with open(base_dir / 'cv_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print()
    print('CV Summary (PoseC3D TAL with Focal Loss):')
    print(f'  Clip Top-1: {summary.get(\"clip_top1_acc_mean\", summary.get(\"top1_acc_mean\", 0)):.2f} +/- {summary.get(\"clip_top1_acc_std\", summary.get(\"top1_acc_std\", 0)):.2f}%')
    print(f'  Mean Acc:   {summary.get(\"mean_class_accuracy_mean\", 0):.2f} +/- {summary.get(\"mean_class_accuracy_std\", 0):.2f}%')
    print(f'  Macro-F1:   {summary.get(\"clip_macro_f1_mean\", 0):.2f} +/- {summary.get(\"clip_macro_f1_std\", 0):.2f}%')
else:
    print('No fold metrics found yet.')
"

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="

