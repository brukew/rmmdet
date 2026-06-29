#!/bin/bash -l
# V-JEPA2 RMM classification with cross-validation.
# Uses best hyperparameters from sweep: f64, lr=1e-5, bs=1, acc=8, ep=20.
#
# Usage:
#   sbatch run_vjepa_cv.sh              # with cropping (default)
#   sbatch --export=ENABLE_CROP=0 run_vjepa_cv.sh  # without cropping
#   sbatch --export=MAX_FOLDS=3 run_vjepa_cv.sh    # limit to 3 folds
#
# Logs: /orcd/data/satra/001/users/brukew/vjepa_rmm_logs

#SBATCH -J vjepa2_cv
#SBATCH -p mit_preemptable
#SBATCH -c 8
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH --requeue
#SBATCH --signal=TERM@120
#SBATCH -o /orcd/data/satra/001/users/brukew/vjepa_rmm_logs/vjepa_cv_4class%j.out
#SBATCH -e /orcd/data/satra/001/users/brukew/vjepa_rmm_logs/vjepa_cv_4class%j.err

set -eo pipefail

if [ -f ~/.bashrc ]; then
  source ~/.bashrc
fi

cd /orcd/data/satra/001/users/brukew
conda activate vjepa2

# Canonical filesystem paths from the repo's single source of truth (config.yaml).
eval "$(python actreg/paths.py --export)"

LOG_DIR=/orcd/data/satra/001/users/brukew/vjepa_rmm_logs
mkdir -p "$LOG_DIR"

# ============================================================================
# Best hyperparameters from sweep
# ============================================================================
FRAMES=${FRAMES:-64}
LR=${LR:-1e-5}
BATCH_SIZE=${BATCH_SIZE:-1}
ACCUMULATION_STEPS=${ACCUMULATION_STEPS:-8}
NUM_EPOCHS=${NUM_EPOCHS:-20}

# ============================================================================
# Paths
# ============================================================================
CSV_DIR=${CSV_DIR:-/orcd/data/satra/001/users/brukew/actreg/dataprep/splits/cv_splits}
CLIPS_ROOT=${CLIPS_ROOT:-$CLASSIFICATION_CLIPS}
OUTPUT_ROOT=${OUTPUT_ROOT:-/orcd/data/satra/001/users/brukew/actreg/v-jepa/runs/vjepa2_rmm_cv}
MODEL_ID=${MODEL_ID:-facebook/vjepa2-vitl-fpc16-256-ssv2}

# Cropping paths
MASK_CACHE_BASE=${MASK_CACHE_BASE:-$CACHE_FOR_TRACKING}
SAM3_PARSED_CSV=${SAM3_PARSED_CSV:-/orcd/data/satra/001/users/brukew/actreg/dataprep/rmm_sam3_parsed.csv}
VIDEO_META_JSON=${VIDEO_META_JSON:-/orcd/data/satra/001/users/brukew/actreg/dataprep/video_meta.json}

# W&B
WANDB_MODE=${WANDB_MODE:-online}
WANDB_PROJECT=${WANDB_PROJECT:-vjepa-rmm}

# ============================================================================
# Build run name and output path
# ============================================================================
suffix="f${FRAMES}_lr${LR}_bs${BATCH_SIZE}_acc${ACCUMULATION_STEPS}_ep${NUM_EPOCHS}"
ENABLE_CROP=${ENABLE_CROP:-1}
if [ "$ENABLE_CROP" = "1" ]; then
  suffix="${suffix}_crop"
  CROP_ARGS="--enable-crop --mask-cache-base $MASK_CACHE_BASE --sam3-parsed-csv $SAM3_PARSED_CSV --video-meta-json $VIDEO_META_JSON"
else
  CROP_ARGS=""
fi

output_path="${OUTPUT_ROOT}/${suffix}"
run_prefix="vjepa2-cv-${suffix}"

# Optional: limit number of folds
FOLD_ARGS=""
if [ -n "$MAX_FOLDS" ]; then
  FOLD_ARGS="--max-folds $MAX_FOLDS"
fi
if [ -n "$START_FOLD" ]; then
  FOLD_ARGS="$FOLD_ARGS --start-fold $START_FOLD"
fi

echo "============================================================"
echo "V-JEPA2 Cross-Validation Training"
echo "============================================================"
echo "Config: frames=$FRAMES, lr=$LR, bs=$BATCH_SIZE, acc=$ACCUMULATION_STEPS, epochs=$NUM_EPOCHS"
echo "Cropping: $ENABLE_CROP"
echo "Output: $output_path"
echo "Max folds: ${MAX_FOLDS:-all}"
echo "Start fold: ${START_FOLD:-0}"
echo "============================================================"

python actreg/v-jepa/finetune_sails_vjepa2_cv_crop.py \
  --csv-dir "$CSV_DIR" \
  --clips-root "$CLIPS_ROOT" \
  --output-root "$output_path" \
  --model-id "$MODEL_ID" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "${NUM_WORKERS:-8}" \
  --num-epochs "$NUM_EPOCHS" \
  --lr "$LR" \
  --accumulation-steps "$ACCUMULATION_STEPS" \
  --log-interval "${LOG_INTERVAL:-50}" \
  --frames-per-clip "$FRAMES" \
  --topk 2 \
  --wandb-mode "$WANDB_MODE" \
  --wandb-project "$WANDB_PROJECT" \
  --run-prefix "$run_prefix" \
  --log-level "${LOG_LEVEL:-INFO}" \
  $CROP_ARGS \
  $FOLD_ARGS \
  ${EXTRA_ARGS:-}

echo "============================================================"
echo "Done! Results saved to: $output_path"
echo "============================================================"

