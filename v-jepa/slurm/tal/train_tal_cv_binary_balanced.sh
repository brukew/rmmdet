#!/bin/bash
#SBATCH --job-name=vjepa2_tal_bin
#SBATCH --output=slurm-logs/vjepa2_tal_cv_binary_%j.out
#SBATCH --error=slurm-logs/vjepa2_tal_cv_binary_%j.err
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00

# ============================================================================
# V-JEPA2 TAL Fine-tuning - Binary Classification (RMM vs Background)
#
# This script trains a 2-class binary classifier:
#   - Class 0: RMM (any of hands_flapping, jumping, rocking, spinning)
#   - Class 1: Background
#
# Features:
#   - Early stopping (patience=5, monitoring val_loss)
#   - Class-balanced sampling:
#     - RMM (class 0): keep all samples (prob=1.0)
#     - Background (class 1): downsample to 10% (prob=0.1)
#   - SAM3-based child cropping
#   - Cross-entropy loss with class weights
#
# Use case:
#   Binary detection of "any RMM behavior" vs "no behavior"
#   This is useful for TAL where the primary task is detecting RMM presence.
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

echo "=============================================="
echo "V-JEPA2 TAL Training: Binary Classification (RMM vs Background)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=============================================="

# Ensure logs directory exists
mkdir -p ${REPO_ROOT}/v-jepa/logs

# Ensure real-time logging
export PYTHONUNBUFFERED=1

if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

cd ${REPO_ROOT}

# Use vjepa2 env (has torch, numpy, pandas, sklearn, matplotlib)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vjepa2

# Canonical filesystem paths from the repo's single source of truth (config.yaml).
eval "$(python paths.py --export)"

# Configuration
CSV_DIR="${REPO_ROOT}/dataprep/tal/splits_cv_4class"
CLIPS_ROOT="$TAL_CLIPS_ROOT"
OUTPUT_ROOT="${REPO_ROOT}/v-jepa/runs/vjepa2_tal_cv_binary_balanced"

# Class probabilities for binary balanced sampling:
# 0: RMM (all RMM types combined) -> 1.0 (keep all)
# 1: Background -> 0.1 (downsample to 10%)
CLASS_PROB='[1.0, 0.1]'

echo ""
echo "Configuration:"
echo "  Mode: Binary Classification (RMM vs Background)"
echo "  CSV Dir: $CSV_DIR"
echo "  Clips Root: $CLIPS_ROOT"
echo "  Output Root: $OUTPUT_ROOT"
echo "  Class Probabilities: $CLASS_PROB"
echo "  Early Stopping Patience: 5"
echo "  Max Epochs: 20"
echo "  Cropping: Enabled"
echo ""

# Run V-JEPA2 TAL fine-tuning with binary classification
python v-jepa/finetune_sails_vjepa2_tal.py \
    --split-mode cv \
    --csv-dir "$CSV_DIR" \
    --clips-root "$CLIPS_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --binary-classification \
    --num-epochs 20 \
    --batch-size 1 \
    --accumulation-steps 4 \
    --lr 1e-5 \
    --num-workers 4 \
    --frames-per-clip 32 \
    --topk 2 \
    --class-prob "$CLASS_PROB" \
    --early-stopping-patience 5 \
    --enable-crop \
    --wandb-project vjepa-tal \
    --run-prefix vjepa2-tal-binary \
    --log-level INFO

# Run CV summary script
echo ""
echo "=============================================="
echo "Computing CV Summary"
echo "=============================================="

python v-jepa/tools/summarize_vjepa_cv_results.py \
    --work-dir "$OUTPUT_ROOT" \
    --task tal \
    --metadata "binary=true" "class_prob=$CLASS_PROB" "patience=5" "epochs=20"

echo ""
echo "=============================================="
echo "Job completed: $(date)"
echo "=============================================="

