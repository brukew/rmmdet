#!/bin/bash -l
# V-JEPA2 RMM classification on single train/val/test split.
# Uses best hyperparameters from sweep: f64, lr=1e-5, bs=1, acc=8, ep=20.
#
# Usage:
#   sbatch run_vjepa_single.sh              # with cropping (default)
#   sbatch --export=ENABLE_CROP=0 run_vjepa_single.sh  # without cropping
#
# Logs: ${REPO_ROOT}/vjepa_rmm_logs

#SBATCH -J vjepa2_single
#SBATCH -p mit_preemptable
#SBATCH -c 8
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH --requeue
#SBATCH --signal=TERM@120
#SBATCH -o slurm-logs/vjepa_single%j.out
#SBATCH -e slurm-logs/vjepa_single%j.err

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

if [ -f ~/.bashrc ]; then
  source ~/.bashrc
fi

cd ${REPO_ROOT}
conda activate vjepa2

# Canonical filesystem paths from the repo's single source of truth (config.yaml).
eval "$(python paths.py --export)"

LOG_DIR=${REPO_ROOT}/vjepa_rmm_logs
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
SPLIT_DIR=${SPLIT_DIR:-${REPO_ROOT}/dataprep/splits/single_split}
CLIPS_ROOT=${CLIPS_ROOT:-$CLASSIFICATION_CLIPS}
CLIP_SUBDIR=${CLIP_SUBDIR:-canonical_clips}  # Use canonical_clips for deduplicated clips
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/v-jepa/runs/vjepa2_rmm_single}
MODEL_ID=${MODEL_ID:-facebook/vjepa2-vitl-fpc16-256-ssv2}

# Cropping paths
MASK_CACHE_BASE=${MASK_CACHE_BASE:-$CACHE_FOR_TRACKING}
SAM3_PARSED_CSV=${SAM3_PARSED_CSV:-${REPO_ROOT}/dataprep/rmm_sam3_parsed.csv}
VIDEO_META_JSON=${VIDEO_META_JSON:-${REPO_ROOT}/dataprep/video_meta.json}

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

output_path="${OUTPUT_DIR}/${suffix}"
run_name="vjepa2-single-${suffix}"

echo "============================================================"
echo "V-JEPA2 Single Split Training"
echo "============================================================"
echo "Config: frames=$FRAMES, lr=$LR, bs=$BATCH_SIZE, acc=$ACCUMULATION_STEPS, epochs=$NUM_EPOCHS"
echo "Cropping: $ENABLE_CROP"
echo "Clips: $CLIPS_ROOT/$CLIP_SUBDIR"
echo "Output: $output_path"
echo "============================================================"

python v-jepa/finetune_sails_vjepa2_single.py \
  --split-dir "$SPLIT_DIR" \
  --clips-root "$CLIPS_ROOT" \
  --clip-subdir "$CLIP_SUBDIR" \
  --output-dir "$output_path" \
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
  --run-name "$run_name" \
  --log-level "${LOG_LEVEL:-INFO}" \
  $CROP_ARGS \
  ${EXTRA_ARGS:-}

echo "============================================================"
echo "Done! Results saved to: $output_path"
echo "============================================================"

