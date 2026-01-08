#!/bin/bash
#SBATCH --job-name=vjepa_feat
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=4:00:00
# Note: Output/error paths are set dynamically via sbatch --output/--error

# =============================================================================
# V-JEPA Feature Extraction for OpenTAD
#
# Environment variables:
#   MODEL_SOURCE: Path to fine-tuned model checkpoint
#   FOLD: CV fold index (0, 1, or 2)
#   ENABLE_CROP: "1" to enable SAM3 child cropping (default: 0)
#   WORKER_INDEX: Index for distributed processing (default: 0)
#   TOTAL_WORKERS: Total number of workers (default: 1)
# =============================================================================

set -eo pipefail

echo "=========================================="
echo "V-JEPA Feature Extraction"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

# Ensure logs directory exists
mkdir -p /orcd/data/satra/001/users/brukew/vjepa_logs

# Activate conda
source ~/.bashrc
conda activate vjepa2

# Configuration from environment
MODEL_SOURCE="${MODEL_SOURCE:-runs/vjepa2_tal_cv_5class_balanced/fold_0}"
FOLD="${FOLD:-0}"
ENABLE_CROP="${ENABLE_CROP:-0}"
WORKER_INDEX="${WORKER_INDEX:-0}"
TOTAL_WORKERS="${TOTAL_WORKERS:-1}"

echo "Model source: ${MODEL_SOURCE}"
echo "Fold: ${FOLD}"
echo "Enable crop: ${ENABLE_CROP}"
echo "Worker: ${WORKER_INDEX}/${TOTAL_WORKERS}"

# Navigate to V-JEPA directory
cd /orcd/data/satra/001/users/brukew/actreg/v-jepa

# Build command
CMD="python tools/extract_vjepa_features.py \
    --model-source ${MODEL_SOURCE} \
    --fold ${FOLD} \
    --worker-index ${WORKER_INDEX} \
    --num-workers ${TOTAL_WORKERS}"

if [ "${ENABLE_CROP}" = "1" ]; then
    CMD="${CMD} --enable-crop"
fi

echo "Running: ${CMD}"
eval ${CMD}

echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="

