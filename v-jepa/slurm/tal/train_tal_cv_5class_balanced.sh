#!/bin/bash
#SBATCH --job-name=vjepa2_tal_bal
#SBATCH --output=/orcd/data/satra/001/users/brukew/actreg/v-jepa/logs/vjepa2_tal_cv_balanced_%j.out
#SBATCH --error=/orcd/data/satra/001/users/brukew/actreg/v-jepa/logs/vjepa2_tal_cv_balanced_%j.err
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00

# ============================================================================
# V-JEPA2 TAL Fine-tuning - Cross-Validation (5 classes: 4 RMM + background)
#
# Features:
#   - Early stopping (patience=5, monitoring val_loss)
#   - Class-balanced sampling:
#     - Background (class 4): downsample to 10%
#     - Rocking (class 2): upsample ~2x to match jumping
#     - Spinning (class 3): upsample ~9x to match jumping
#   - SAM3-based child cropping
#   - Cross-entropy loss with class weights
#
# Class distribution in training set:
#   0: hands_flapping (~500)
#   1: jumping (~237)
#   2: rocking (~123) -> upsample 1.93x
#   3: spinning (~26)  -> upsample 9.12x
#   4: background (~7373) -> downsample to 10%
# ============================================================================

set -eo pipefail

echo "=============================================="
echo "V-JEPA2 TAL Training: 5-class CV with Balanced Sampling"
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

# Configuration
CSV_DIR="/orcd/data/satra/001/users/brukew/actreg/dataprep/tal/splits_cv_4class"
CLIPS_ROOT="/orcd/scratch/bcs/001/brukew/sails/tal_windows_4class/canonical_clips"
OUTPUT_ROOT="/orcd/data/satra/001/users/brukew/actreg/v-jepa/runs/vjepa2_tal_cv_5class_balanced"

# Class probabilities for balanced sampling:
# 0: hands_flapping -> 1.0 (keep all)
# 1: jumping -> 1.0 (keep all, reference for upsampling)
# 2: rocking -> 1.93 (upsample to match jumping: 237/123)
# 3: spinning -> 9.12 (upsample to match jumping: 237/26)
# 4: background -> 0.1 (downsample to 10%)
CLASS_PROB='[1.0, 1.0, 1.93, 9.12, 0.1]'

echo ""
echo "Configuration:"
echo "  CSV Dir: $CSV_DIR"
echo "  Clips Root: $CLIPS_ROOT"
echo "  Output Root: $OUTPUT_ROOT"
echo "  Class Probabilities: $CLASS_PROB"
echo "  Early Stopping Patience: 5"
echo "  Max Epochs: 20"
echo "  Cropping: Enabled"
echo ""

# Run V-JEPA2 TAL fine-tuning with balanced sampling and early stopping
python v-jepa/finetune_sails_vjepa2_tal.py \
    --split-mode cv \
    --csv-dir "$CSV_DIR" \
    --clips-root "$CLIPS_ROOT" \
    --output-root "$OUTPUT_ROOT" \
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
    --run-prefix vjepa2-tal-cv-balanced \
    --log-level INFO

# Run CV summary script
echo ""
echo "=============================================="
echo "Computing CV Summary"
echo "=============================================="

python v-jepa/tools/summarize_vjepa_cv_results.py \
    --work-dir "$OUTPUT_ROOT" \
    --task tal \
    --metadata "class_prob=$CLASS_PROB" "patience=5" "epochs=20"

echo ""
echo "=============================================="
echo "Job completed: $(date)"
echo "=============================================="


