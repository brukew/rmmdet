#!/bin/bash
#SBATCH --job-name=posec3d_tal_ce_bal
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/posec3d_tal_5class_cv_ce_balanced_%j.out
#SBATCH --error=slurm-logs/posec3d_tal_5class_cv_ce_balanced_%j.err

# ============================================================================
# PoseC3D TAL Training: 5-class CV with CE Loss + Balanced Classes
# 
# Features:
#   - Cross-Entropy loss (not Focal)
#   - Inverse-frequency class weights
#   - Background subsampling (10%) - more aggressive
#   - Upsampling rocking (1.93x) and spinning (9.12x) to match jumping
#   - Early stopping (patience=5) - more patient than default
#
# Class distribution after balancing:
#   hands_flapping: 500 -> 500 (1.0x)
#   jumping: 237 -> 237 (1.0x) - reference
#   rocking: 123 -> ~237 (1.93x) - upsample
#   spinning: 26 -> ~237 (9.12x) - upsample
#   background: 7373 -> ~737 (0.1x) - downsample
#
# Runs on all 3 folds.
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
echo "PoseC3D TAL Training: 5-class CV with CE + Balanced Classes"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

# Ensure logs directory exists
mkdir -p ${REPO_ROOT}/pyskl_logs/posec3d/tal

# Ensure real-time logging
export PYTHONUNBUFFERED=1

if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyskl

cd ${REPO_ROOT}/pyskl

# Configuration
CONFIG="configs/posec3d/slowonly_r50_sails_k400p/joint_tal_5class.py"
BASE_ANN_DIR="data/sails/tal/cv_4class/5class_windows_conf04"
BASE_WORK_DIR="work_dirs/posec3d/tal/cv_4class_5class_ce_balanced"
EPOCHS=24
LR=0.00125

# Class balancing via class_prob:
# [hands_flapping, jumping, rocking, spinning, background]
# Target: match jumping (237 samples)
# rocking: 237/123 = 1.93
# spinning: 237/26 = 9.12
# background: 0.1 (10% subsample)
CLASS_PROB="[1.0,1.0,1.93,9.12,0.1]"

echo ""
echo "Config: $CONFIG"
echo "Annotation dir: $BASE_ANN_DIR"
echo "Work dir: $BASE_WORK_DIR"
echo "Epochs: $EPOCHS (with early stopping patience=5)"
echo "Learning rate: $LR"
echo "Class prob: $CLASS_PROB"
echo "Loss: Cross-Entropy + inverse class weights"
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
    
    # Train with CE loss, class weights, and balanced sampling
    # Use --class-prob for upsampling minority classes and downsampling background
    python tools/train_weighted.py $CONFIG \
        --ann-file $ANN_FILE \
        --work-dir $WORK_DIR \
        --total-epochs $EPOCHS \
        --lr $LR \
        --class-prob "$CLASS_PROB" \
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
        --task tal \
        --cfg-options \
            data.train.dataset.ann_file=$ANN_FILE \
            data.val.ann_file=$ANN_FILE \
            model.cls_head.num_classes=5
done

# Compute CV Summary using standalone script
python tools/summarize_cv_results.py \
    --work-dir $BASE_WORK_DIR \
    --model posec3d \
    --task tal \
    --loss ce \
    --metadata "class_prob=$CLASS_PROB" "patience=5"

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="

