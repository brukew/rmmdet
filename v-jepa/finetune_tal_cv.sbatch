#!/bin/bash
#SBATCH --job-name=vjepa2_tal_cv
#SBATCH --output=/orcd/data/satra/001/users/brukew/actreg/v-jepa/logs/vjepa2_tal_cv_%j.out
#SBATCH --error=/orcd/data/satra/001/users/brukew/actreg/v-jepa/logs/vjepa2_tal_cv_%j.err
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# ============================================================================
# V-JEPA2 TAL Fine-tuning - Cross-Validation (5 classes: 4 RMM + background)
# ============================================================================

set -eo pipefail

echo "=============================================="
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
conda activate vjepa2

# Run V-JEPA2 TAL fine-tuning
python v-jepa/finetune_sails_vjepa2_tal.py \
    --split-mode cv \
    --csv-dir /orcd/data/satra/001/users/brukew/actreg/dataprep/tal/splits_cv_4class \
    --clips-root /orcd/scratch/bcs/001/brukew/sails/tal_windows_4class/canonical_clips \
    --output-root /orcd/data/satra/001/users/brukew/actreg/v-jepa/runs/vjepa2_tal_cv_5class \
    --num-epochs 20 \
    --batch-size 1 \
    --accumulation-steps 4 \
    --lr 1e-5 \
    --num-workers 8 \
    --frames-per-clip 32 \
    --topk 2 \
    --enable-crop \
    --wandb-project vjepa-tal \
    --run-prefix vjepa2-tal-cv \
    --log-level INFO

echo "=============================================="
echo "Job completed: $(date)"
echo "=============================================="

