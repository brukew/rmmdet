#!/bin/bash
#
# Submit distributed feature extraction jobs across multiple SLURM nodes.
#
# Usage:
#   ./extract_distributed.sh \
#     --model-source runs/vjepa2_tal_cv_5class_balanced/fold_0 \
#     --fold 0 \
#     --num-workers 4 \
#     --enable-crop 1
#

set -e

# Default values
MODEL_SOURCE=""
FOLD=""
NUM_WORKERS=4
PARTITIONS="pi_satra,mit_preemptable,mit_normal_gpu"  # Cycle through these
ENABLE_CROP="1"
RESUME="0"
LOG_DIR="/orcd/data/satra/001/users/brukew/vjepa_logs"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-source)
            MODEL_SOURCE="$2"
            shift 2
            ;;
        --fold)
            FOLD="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --partitions)
            PARTITIONS="$2"
            shift 2
            ;;
        --enable-crop)
            ENABLE_CROP="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL_SOURCE" ] || [ -z "$FOLD" ]; then
    echo "Usage: $0 --model-source <path> --fold <n> [--num-workers <n>] [--enable-crop 1]"
    exit 1
fi

# Convert relative path to absolute
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "${SCRIPT_DIR}")"

if [[ ! "$MODEL_SOURCE" = /* ]]; then
    MODEL_SOURCE="${BASE_DIR}/${MODEL_SOURCE}"
fi

# Extract model name for log naming
MODEL_NAME=$(basename "$(dirname "${MODEL_SOURCE}")")
FOLD_NAME=$(basename "${MODEL_SOURCE}")

echo "=============================================="
echo "Distributed V-JEPA Feature Extraction"
echo "=============================================="
echo "Model source: ${MODEL_SOURCE}"
echo "Model name: ${MODEL_NAME}"
echo "Fold: ${FOLD}"
echo "Num workers: ${NUM_WORKERS}"
echo "Partitions: ${PARTITIONS}"
echo "Enable crop: ${ENABLE_CROP}"
echo ""

# Parse partitions into array
IFS=',' read -ra PARTITION_ARR <<< "$PARTITIONS"
NUM_PARTITIONS=${#PARTITION_ARR[@]}

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Submit jobs for each worker, cycling through partitions
JOB_IDS=""
for ((i=0; i<NUM_WORKERS; i++)); do
    # Cycle through partitions
    PARTITION_IDX=$((i % NUM_PARTITIONS))
    PARTITION="${PARTITION_ARR[$PARTITION_IDX]}"
    
    # Create descriptive log names: model_fold_workerN_jobid
    LOG_PREFIX="${LOG_DIR}/${MODEL_NAME}_f${FOLD}_w${i}"
    
    echo "Submitting worker ${i}/${NUM_WORKERS} on ${PARTITION}..."
    
    JOB_ID=$(sbatch \
        --partition=${PARTITION} \
        --output="${LOG_PREFIX}_%j.out" \
        --error="${LOG_PREFIX}_%j.err" \
        --export=MODEL_SOURCE="${MODEL_SOURCE}",FOLD="${FOLD}",ENABLE_CROP="${ENABLE_CROP}",WORKER_INDEX="${i}",TOTAL_WORKERS="${NUM_WORKERS}" \
        "${BASE_DIR}/slurm/extract_features.sh" \
        | awk '{print $NF}')
    
    echo "  Submitted job: ${JOB_ID} -> ${LOG_PREFIX}_${JOB_ID}.{out,err}"
    
    if [ -n "$JOB_IDS" ]; then
        JOB_IDS="${JOB_IDS}:${JOB_ID}"
    else
        JOB_IDS="${JOB_ID}"
    fi
done

echo ""
echo "=============================================="
echo "All jobs submitted!"
echo "Job IDs: ${JOB_IDS}"
echo ""
echo "Monitor with: squeue -u $USER"
echo "=============================================="

