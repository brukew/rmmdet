#!/bin/bash
#SBATCH --job-name=stgcnpp_4cls_cv_b
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-logs/4class_cv_b_%j.out
#SBATCH --error=slurm-logs/4class_cv_b_%j.err

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

echo "=========================================="
echo "STGCN++ Training: 4-class 3-Fold CV (Bone)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

# Setup environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyskl

cd ${REPO_ROOT}/pyskl

# Configuration
CONFIG="configs/stgcn++/stgcnpp_sails_ntu60p/b.py"
BASE_WORK_DIR="work_dirs/stgcnpp/cv/4class_conf04_b"
EPOCHS=24
LR=0.01

mkdir -p $BASE_WORK_DIR

# Train each fold sequentially
for FOLD in 0 1 2; do
    echo ""
    echo "=========================================="
    echo "Training Fold $FOLD"
    echo "=========================================="
    
    ANN_FILE="data/sails/cv/4class_conf04/fold${FOLD}.pkl"
    WORK_DIR="${BASE_WORK_DIR}/fold${FOLD}"
    
    echo "Config: $CONFIG"
    echo "Annotation: $ANN_FILE"
    echo "Work dir: $WORK_DIR"
    
    # Train
    python tools/train.py $CONFIG \
        --work-dir $WORK_DIR \
        --cfg-options \
            data.train.dataset.ann_file=$ANN_FILE \
            data.val.ann_file=$ANN_FILE \
            total_epochs=$EPOCHS \
            optimizer.lr=$LR \
        --validate \
        --launcher none
    
    # Evaluate on val set
    BEST_CKPT=$(ls -t ${WORK_DIR}/best_*.pth 2>/dev/null | head -1)
    if [ -z "$BEST_CKPT" ]; then
        BEST_CKPT="${WORK_DIR}/latest.pth"
    fi
    
    echo "Evaluating fold $FOLD with: $BEST_CKPT"
    
    python tools/evaluate_sails.py $CONFIG \
        -C $BEST_CKPT \
        --split val \
        --output-dir ${WORK_DIR}/eval_val \
        --cfg-options \
            data.train.dataset.ann_file=$ANN_FILE \
            data.val.ann_file=$ANN_FILE
done

# Aggregate CV results
echo ""
echo "=========================================="
echo "Aggregating CV Results"
echo "=========================================="

python -c "
import json
import numpy as np
from pathlib import Path

base_dir = Path('${BASE_WORK_DIR}')
metrics_keys = ['clip_top1', 'clip_top2', 'clip_macro_f1', 'clip_weighted_f1', 
                'clip_macro_precision', 'clip_macro_recall', 'clip_kappa',
                'video_top1', 'video_top2', 'video_macro_f1', 'video_weighted_f1',
                'video_macro_precision', 'video_macro_recall', 'video_kappa']

all_metrics = {k: [] for k in metrics_keys}
fold_metrics = []

for fold in range(3):
    metrics_file = base_dir / f'fold{fold}' / 'eval_val' / 'metrics.json'
    if metrics_file.exists():
        with open(metrics_file) as f:
            m = json.load(f)
            fold_metrics.append(m)
            for k in metrics_keys:
                if k in m:
                    all_metrics[k].append(m[k])

# Calculate mean and std
summary = {'per_fold': fold_metrics}
for k, vals in all_metrics.items():
    if vals:
        summary[f'{k}_mean'] = float(np.mean(vals))
        summary[f'{k}_std'] = float(np.std(vals))

with open(base_dir / 'cv_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('CV Summary:')
print(f\"  Clip Top-1: {summary.get('clip_top1_mean', 0):.2%} ± {summary.get('clip_top1_std', 0):.2%}\")
print(f\"  Clip Macro F1: {summary.get('clip_macro_f1_mean', 0):.2%} ± {summary.get('clip_macro_f1_std', 0):.2%}\")
print(f\"  Video Top-1: {summary.get('video_top1_mean', 0):.2%} ± {summary.get('video_top1_std', 0):.2%}\")
print(f\"  Video Macro F1: {summary.get('video_macro_f1_mean', 0):.2%} ± {summary.get('video_macro_f1_std', 0):.2%}\")
"

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="

