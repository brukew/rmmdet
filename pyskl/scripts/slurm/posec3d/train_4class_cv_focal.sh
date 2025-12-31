#!/bin/bash
#SBATCH --job-name=posec3d_4cls_cv_focal
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/orcd/data/satra/001/users/brukew/pyskl_logs/posec3d/4class_cv_focal_%j.out
#SBATCH --error=/orcd/data/satra/001/users/brukew/pyskl_logs/posec3d/4class_cv_focal_%j.err

echo "=========================================="
echo "PoseC3D Training: 4-class CV WITH FOCAL LOSS + EARLY STOPPING"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyskl

cd /orcd/data/satra/001/users/brukew/actreg/pyskl

# Configuration - using focal loss config
CONFIG="configs/posec3d/slowonly_r50_sails_k400p/joint_focal.py"
BASE_ANN_DIR="data/sails/cv/4class_conf04"
BASE_WORK_DIR="work_dirs/posec3d/cv/4class_conf04_focal"
EPOCHS=24  # More epochs since early stopping will cut it short if needed
LR=0.00125
# Class names - use default (don't pass --class-names for 4-class)

# 3-fold CV
for FOLD in 0 1 2; do
    ANN_FILE="${BASE_ANN_DIR}/fold${FOLD}.pkl"
    WORK_DIR="${BASE_WORK_DIR}/fold${FOLD}"
    
    echo ""
    echo "######################################################"
    echo "# FOLD ${FOLD}"
    echo "######################################################"
    echo "Annotation: $ANN_FILE"
    echo "Work dir: $WORK_DIR"
    echo "Config: $CONFIG (Focal Loss + Early Stopping patience=2)"
    
    # Train with focal loss (early stopping is configured in config)
    python tools/train.py $CONFIG \
        --work-dir $WORK_DIR \
        --validate \
        --launcher none \
        --cfg-options \
            data.train.dataset.ann_file=$ANN_FILE \
            data.val.ann_file=$ANN_FILE \
            total_epochs=$EPOCHS \
            optimizer.lr=$LR
    
    # Evaluate on val set
    BEST_CKPT=$(ls -t ${WORK_DIR}/best_*.pth 2>/dev/null | head -1)
    if [ -z "$BEST_CKPT" ]; then
        BEST_CKPT="${WORK_DIR}/latest.pth"
    fi
    
    echo "Evaluating fold ${FOLD}: $BEST_CKPT"
    
    python tools/evaluate_sails.py $CONFIG \
        -C $BEST_CKPT \
        --split val \
        --output-dir ${WORK_DIR}/eval_val \
        --cfg-options \
            data.train.dataset.ann_file=$ANN_FILE \
            data.val.ann_file=$ANN_FILE
done

# CV Summary
echo ""
echo "=========================================="
echo "Computing CV Summary"
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

if folds:
    summary = {'per_fold': folds}
    metric_keys = [k for k in folds[0].keys() if k != 'fold']
    for key in metric_keys:
        vals = [f[key] for f in folds if key in f]
        if vals and isinstance(vals[0], (int, float)):
            summary[f'{key}_mean'] = float(np.mean(vals))
            summary[f'{key}_std'] = float(np.std(vals))
    
    with open(base_dir / 'cv_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print('CV Summary (Focal Loss + Early Stopping):')
    print(f'  Clip Top-1: {summary.get(\"clip_top1_acc_mean\", summary.get(\"top1_acc_mean\", 0)):.2f} ± {summary.get(\"clip_top1_acc_std\", summary.get(\"top1_acc_std\", 0)):.2f}%')
    print(f'  Video Top-1: {summary.get(\"video_top1_acc_mean\", 0):.2f} ± {summary.get(\"video_top1_acc_std\", 0):.2f}%')
    print(f'  Clip Macro-F1: {summary.get(\"clip_macro_f1_mean\", 0):.2f} ± {summary.get(\"clip_macro_f1_std\", 0):.2f}%')
    print(f'  Video Macro-F1: {summary.get(\"video_macro_f1_mean\", 0):.2f} ± {summary.get(\"video_macro_f1_std\", 0):.2f}%')
else:
    print('No fold metrics found yet.')
"

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="

