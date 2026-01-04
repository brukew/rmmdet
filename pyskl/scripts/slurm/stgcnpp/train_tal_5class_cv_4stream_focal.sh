#!/bin/bash
#SBATCH --job-name=stgcnpp_tal_focal
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/orcd/data/satra/001/users/brukew/pyskl_logs/stgcnpp/tal/stgcnpp_tal_5class_cv_4stream_focal_%j.out
#SBATCH --error=/orcd/data/satra/001/users/brukew/pyskl_logs/stgcnpp/tal/stgcnpp_tal_5class_cv_4stream_focal_%j.err

# ============================================================================
# STGCN++ TAL Training: 5-class CV 4-Stream with Focal Loss + Early Stopping
# 
# Features:
#   - Focal loss (gamma=2) for hard example mining
#   - Inverse-frequency class weights (computed automatically)
#   - Background subsampling (20%) to reduce class imbalance
#   - Early stopping (patience=3) to prevent overfitting
#   - All 4 modalities: joint (j), bone (b), joint motion (jm), bone motion (bm)
#
# Runs on all 3 folds for each modality.
# ============================================================================

set -eo pipefail

echo "=========================================="
echo "STGCN++ TAL Training: 5-class CV 4-Stream with Focal Loss"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

# Ensure logs directory exists
mkdir -p /orcd/data/satra/001/users/brukew/pyskl_logs/stgcnpp/tal

# Ensure real-time logging
export PYTHONUNBUFFERED=1

if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyskl

cd /orcd/data/satra/001/users/brukew/actreg/pyskl

# Configuration
CONFIG_DIR="configs/stgcn++/stgcnpp_sails_ntu60p"
BASE_ANN_DIR="data/sails/tal/cv_4class/5class_windows_conf04"
BASE_WORK_DIR="work_dirs/stgcnpp/tal/cv_4class_5class_focal"
EPOCHS=24
BG_SUBSAMPLE=0.2

# Modalities
MODALITIES="j b jm bm"

echo ""
echo "Config dir: $CONFIG_DIR"
echo "Annotation dir: $BASE_ANN_DIR"
echo "Work dir: $BASE_WORK_DIR"
echo "Epochs: $EPOCHS (with early stopping patience=3)"
echo "Background subsample: $BG_SUBSAMPLE"
echo "Loss: Focal Loss (gamma=2) + inverse class weights"
echo "Modalities: $MODALITIES"
echo ""

# Train each modality across all 3 folds
for MODALITY in $MODALITIES; do
    CONFIG="${CONFIG_DIR}/${MODALITY}_tal_5class_focal.py"
    
    echo ""
    echo "======================================================"
    echo "MODALITY: $MODALITY"
    echo "======================================================"
    
    for FOLD in 0 1 2; do
        ANN_FILE="${BASE_ANN_DIR}/fold${FOLD}.pkl"
        WORK_DIR="${BASE_WORK_DIR}/${MODALITY}/fold${FOLD}"
        
        echo ""
        echo "------------------------------------------------------"
        echo "MODALITY: $MODALITY | FOLD: $FOLD"
        echo "------------------------------------------------------"
        echo "Config: $CONFIG"
        echo "Annotation: $ANN_FILE"
        echo "Work dir: $WORK_DIR"
        
        # Check if pickle exists
        if [ ! -f "$ANN_FILE" ]; then
            echo "ERROR: Annotation file not found: $ANN_FILE"
            continue
        fi
        
        # Train with focal loss, class weights, and background subsampling
        python tools/train_weighted.py $CONFIG \
            --ann-file $ANN_FILE \
            --work-dir $WORK_DIR \
            --total-epochs $EPOCHS \
            --bg-subsample $BG_SUBSAMPLE \
            --validate \
            --launcher none
        
        # Evaluate on val set
        BEST_CKPT=$(ls -t ${WORK_DIR}/best_*.pth 2>/dev/null | head -1)
        if [ -z "$BEST_CKPT" ]; then
            BEST_CKPT="${WORK_DIR}/latest.pth"
        fi
        
        echo ""
        echo "Evaluating $MODALITY fold ${FOLD}: $BEST_CKPT"
        
        python tools/evaluate_sails.py $CONFIG \
            -C $BEST_CKPT \
            --split val \
            --output-dir ${WORK_DIR}/eval_val \
            --cfg-options \
                data.train.dataset.ann_file=$ANN_FILE \
                data.val.ann_file=$ANN_FILE \
                model.cls_head.num_classes=5
    done
done

# Per-modality CV Summary
echo ""
echo "=========================================="
echo "Computing Per-Modality CV Summaries"
echo "=========================================="

for MODALITY in $MODALITIES; do
    python -c "
import json
import numpy as np
from pathlib import Path

modality = '${MODALITY}'
base_dir = Path('${BASE_WORK_DIR}') / modality
folds = []

for fold in range(3):
    metrics_file = base_dir / f'fold{fold}' / 'eval_val' / 'metrics.json'
    if metrics_file.exists():
        with open(metrics_file) as f:
            m = json.load(f)
            m['fold'] = fold
            folds.append(m)
            print(f'{modality} Fold {fold}: loaded')
    else:
        print(f'{modality} Fold {fold}: not found')

if folds:
    summary = {'modality': modality, 'per_fold': folds}
    metric_keys = [k for k in folds[0].keys() if k != 'fold']
    for key in metric_keys:
        vals = [f[key] for f in folds if key in f]
        if vals and isinstance(vals[0], (int, float)):
            summary[f'{key}_mean'] = float(np.mean(vals))
            summary[f'{key}_std'] = float(np.std(vals))
    
    with open(base_dir / 'cv_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f'{modality} CV: top1={summary.get(\"clip_top1_acc_mean\", summary.get(\"top1_acc_mean\", 0)):.2f}%')
"
done

# 4-Stream Fusion Summary
echo ""
echo "=========================================="
echo "Computing 4-Stream Fusion Summary"
echo "=========================================="

python -c "
import json
import numpy as np
from pathlib import Path

base_dir = Path('${BASE_WORK_DIR}')
modalities = ['j', 'b', 'jm', 'bm']
all_results = {}

for mod in modalities:
    summary_file = base_dir / mod / 'cv_summary.json'
    if summary_file.exists():
        with open(summary_file) as f:
            all_results[mod] = json.load(f)

if all_results:
    fusion_summary = {
        'model': 'STGCN++',
        'task': 'TAL 5-class',
        'loss': 'focal',
        'bg_subsample': 0.2,
        'per_modality': {}
    }
    
    for mod, data in all_results.items():
        fusion_summary['per_modality'][mod] = {
            'top1_acc_mean': data.get('clip_top1_acc_mean', data.get('top1_acc_mean', 0)),
            'top1_acc_std': data.get('clip_top1_acc_std', data.get('top1_acc_std', 0)),
        }
    
    # Compute ensemble average
    top1_means = [v['top1_acc_mean'] for v in fusion_summary['per_modality'].values()]
    fusion_summary['ensemble_avg_top1'] = float(np.mean(top1_means))
    
    with open(base_dir / 'fusion_summary.json', 'w') as f:
        json.dump(fusion_summary, f, indent=2)
    
    print('4-Stream Results:')
    for mod, v in fusion_summary['per_modality'].items():
        print(f'  {mod}: {v[\"top1_acc_mean\"]:.2f} +/- {v[\"top1_acc_std\"]:.2f}%')
    print(f'  Ensemble Avg: {fusion_summary[\"ensemble_avg_top1\"]:.2f}%')
"

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="

