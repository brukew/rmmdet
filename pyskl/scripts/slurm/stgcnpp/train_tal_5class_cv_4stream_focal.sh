#!/bin/bash
#SBATCH --job-name=stgcnpp_tal_focal
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-logs/stgcnpp_tal_5class_cv_4stream_focal_%j.out
#SBATCH --error=slurm-logs/stgcnpp_tal_5class_cv_4stream_focal_%j.err

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
echo "STGCN++ TAL Training: 5-class CV 4-Stream with Focal Loss"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

# Ensure logs directory exists
mkdir -p ${REPO_ROOT}/pyskl_logs/stgcnpp/tal

# Ensure real-time logging
export PYTHONUNBUFFERED=1

if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyskl

cd ${REPO_ROOT}/pyskl

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
            --task tal \
            --cfg-options \
                data.train.dataset.ann_file=$ANN_FILE \
                data.val.ann_file=$ANN_FILE \
                model.cls_head.num_classes=5
    done
done

# Compute CV Summary using standalone script
python tools/summarize_cv_results.py \
    --work-dir $BASE_WORK_DIR \
    --model stgcnpp \
    --task tal \
    --loss focal \
    --modalities $MODALITIES \
    --metadata "bg_subsample=$BG_SUBSAMPLE" "patience=3"

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="

