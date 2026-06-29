#!/bin/bash -l
# Hyperparameter sweep for V-JEPA2 RMM classification with cropping.
# Array size and max concurrency are set for mit_preemptable (4 concurrent).
# If you run on pi_satra, change the array limit to %2.
#
# Logs: /orcd/data/satra/001/users/brukew/vjepa_rmm_logs

#SBATCH -J vjepa2_sweep
#SBATCH -p mit_preemptable
#SBATCH -c 8
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH --requeue
#SBATCH --signal=TERM@120
#SBATCH -o /orcd/data/satra/001/users/brukew/vjepa_rmm_logs/vjepa_sweep_%A_%a.out
#SBATCH -e /orcd/data/satra/001/users/brukew/vjepa_rmm_logs/vjepa_sweep_%A_%a.err
#SBATCH -a 0-7%4

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

# Grid of 16 configs (frames, lr, batch size, accumulation steps, epochs)
CONFIGS=(
  # "frames=16 lr=1e-5   bs=1 acc=8  epochs=20"
  # "frames=16 lr=2e-5   bs=1 acc=8  epochs=20"
  # "frames=16 lr=1e-5   bs=2 acc=8  epochs=20"
  # "frames=16 lr=1e-5   bs=2 acc=4  epochs=20"
  # "frames=16 lr=5e-6   bs=2 acc=8  epochs=20"
  # "frames=32 lr=1e-5   bs=1 acc=8  epochs=20"
  "frames=32 lr=1.5e-5 bs=1 acc=8  epochs=20"
  "frames=32 lr=5e-6   bs=1 acc=8  epochs=20"
  "frames=32 lr=1e-5   bs=2 acc=8  epochs=20"
  "frames=32 lr=1e-5   bs=2 acc=4  epochs=20"
  "frames=32 lr=1e-5   bs=2 acc=4  epochs=20"
  "frames=64 lr=5e-6   bs=1 acc=8  epochs=20"
  "frames=64 lr=1e-5   bs=1 acc=8  epochs=20"
  "frames=64 lr=5e-6   bs=1 acc=4  epochs=20"
  # "frames=64 lr=5e-6   bs=2 acc=8  epochs=20"
  # "frames=64 lr=1e-5   bs=2 acc=8  epochs=20"
)

idx=${SLURM_ARRAY_TASK_ID:-0}
if [ "$idx" -lt 0 ] || [ "$idx" -ge "${#CONFIGS[@]}" ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=$idx (max index $(( ${#CONFIGS[@]} - 1 )))" >&2
  exit 1
fi

eval "${CONFIGS[$idx]}"

CSV_DIR=${CSV_DIR:-/orcd/data/satra/001/users/brukew/actreg/dataprep/cv_folds}
CLIPS_ROOT=${CLIPS_ROOT:-$VJEPA2_FINETUNE_CLIPS}
OUTPUT_ROOT=${OUTPUT_ROOT:-/orcd/data/satra/001/users/brukew/actreg/v-jepa/runs/vjepa2_rmm_type_cv_sweep}
MODEL_ID=${MODEL_ID:-facebook/vjepa2-vitl-fpc16-256-ssv2}
MASK_CACHE_BASE=${MASK_CACHE_BASE:-$CACHE_FOR_TRACKING}
SAM3_PARSED_CSV=${SAM3_PARSED_CSV:-/orcd/data/satra/001/users/brukew/actreg/dataprep/rmm_sam3_parsed.csv}
VIDEO_META_JSON=${VIDEO_META_JSON:-/orcd/data/satra/001/users/brukew/actreg/dataprep/video_meta.json}

RUN_PREFIX=${RUN_PREFIX:-vjepa2-rmm-cv-sweep}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_PROJECT=${WANDB_PROJECT:-vjepa-rmm}

suffix="f${frames}_lr${lr}_bs${bs}_acc${acc}_ep${epochs}"
fold_out="${OUTPUT_ROOT}/${suffix}"

python actreg/v-jepa/finetune_sails_vjepa2_cv_crop.py \
  --csv-dir "$CSV_DIR" \
  --clips-root "$CLIPS_ROOT" \
  --output-root "$fold_out" \
  --model-id "$MODEL_ID" \
  --batch-size "$bs" \
  --num-workers "${NUM_WORKERS:-8}" \
  --num-epochs "$epochs" \
  --lr "$lr" \
  --accumulation-steps "$acc" \
  --log-interval "${LOG_INTERVAL:-50}" \
  --frames-per-clip "$frames" \
  --topk 2 \
  --start-fold "${START_FOLD:-0}" \
  --wandb-mode "$WANDB_MODE" \
  --wandb-project "$WANDB_PROJECT" \
  --run-prefix "f$RUN_PREFIX_lr${lr}_bs${bs}_acc${acc}_ep${epochs}" \
  --log-level "${LOG_LEVEL:-INFO}" \
  --enable-crop \
  --mask-cache-base "$MASK_CACHE_BASE" \
  --sam3-parsed-csv "$SAM3_PARSED_CSV" \
  --video-meta-json "$VIDEO_META_JSON" \
  ${EXTRA_ARGS:-}
