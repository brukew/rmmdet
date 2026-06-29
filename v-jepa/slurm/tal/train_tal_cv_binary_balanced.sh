#!/bin/bash
#SBATCH --job-name=vjepa2_tal_bin
#SBATCH --output=/orcd/data/satra/001/users/brukew/actreg/v-jepa/logs/vjepa2_tal_cv_binary_%j.out
#SBATCH --error=/orcd/data/satra/001/users/brukew/actreg/v-jepa/logs/vjepa2_tal_cv_binary_%j.err
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

set -eo pipefail

echo "=============================================="
echo "V-JEPA2 TAL Training: Binary Classification (RMM vs Background)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=============================================="

# Ensure logs directory exists
mkdir -p /orcd/data/satra/001/users/brukew/actreg/v-jepa/logs

# Ensure real-time logging
export PYTHONUNBUFFERED=1

if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

cd /orcd/data/satra/001/users/brukew/actreg

# Use vjepa2 env (has torch, numpy, pandas, sklearn, matplotlib)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vjepa2

# Canonical filesystem paths from the repo's single source of truth (config.yaml).
eval "$(python paths.py --export)"

# Configuration
CSV_DIR="/orcd/data/satra/001/users/brukew/actreg/dataprep/tal/splits_cv_4class"
CLIPS_ROOT="$TAL_CLIPS_ROOT"
OUTPUT_ROOT="/orcd/data/satra/001/users/brukew/actreg/v-jepa/runs/vjepa2_tal_cv_binary_balanced"

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

