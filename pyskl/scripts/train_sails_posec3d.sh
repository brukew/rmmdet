#!/bin/bash
#
# Train PoseC3D on SAILS dataset with K400 pretrained weights.
#
# Usage:
#   ./scripts/train_sails_posec3d.sh [NUM_GPUS] [--options]
#
# Examples:
#   ./scripts/train_sails_posec3d.sh 1                    # Single GPU
#   ./scripts/train_sails_posec3d.sh 4 --validate         # 4 GPUs with validation
#   ./scripts/train_sails_posec3d.sh 8 --validate --test-best  # Full training
#

set -e

# Default to 1 GPU if not specified
NUM_GPUS=${1:-1}
shift || true

CONFIG="configs/posec3d/slowonly_r50_sails_k400p/joint.py"

echo "=============================================="
echo "Training PoseC3D on SAILS dataset"
echo "Config: ${CONFIG}"
echo "GPUs: ${NUM_GPUS}"
echo "Additional args: $@"
echo "=============================================="

cd "$(dirname "$0")/.."

if [ ${NUM_GPUS} -eq 1 ]; then
    echo "Running single-GPU training..."
    python tools/train.py ${CONFIG} "$@"
else
    echo "Running distributed training with ${NUM_GPUS} GPUs..."
    bash tools/dist_train.sh ${CONFIG} ${NUM_GPUS} "$@"
fi

echo "Training complete!"








