#!/bin/bash -l
#SBATCH -J vjepa2_cv
#SBATCH -p mit_preemptable
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:h100:1
#SBATCH -t 48:00:00
#SBATCH --requeue
#SBATCH --signal=TERM@120
#SBATCH -o slurm-logs/vjepa2_cv_%j.out
#SBATCH -e slurm-logs/vjepa2_cv_%j.err

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

conda activate vjepa2

# Optional: export CONDA_ENV to activate a specific environment (e.g., vjepa2)
# if [ -n "$CONDA_ENV" ]; then
#     # shellcheck source=/dev/null
#     source ~/.bashrc
#     conda activate vjepa2
# fi

cd ${REPO_ROOT}

# Canonical filesystem paths from the repo's single source of truth (config.yaml).
eval "$(python paths.py --export)"

CSV_DIR=${CSV_DIR:-${REPO_ROOT}/dataprep/cv_folds}
CLIPS_ROOT=${CLIPS_ROOT:-$VJEPA2_FINETUNE_CLIPS}
OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/v-jepa/runs/vjepa2_rmm_type_cv}
MODEL_ID=${MODEL_ID:-facebook/vjepa2-vitl-fpc16-256-ssv2}
WANDB_MODE=${WANDB_MODE:-offline}
WANDB_PROJECT=${WANDB_PROJECT:-vjepa-rmm}
RUN_PREFIX=${RUN_PREFIX:-vjepa2-rmm-cv}
LOG_LEVEL=${LOG_LEVEL:-INFO}
START_FOLD=${START_FOLD:-0}
REUSE_CHECKPOINTS=${REUSE_CHECKPOINTS:-0}

MAX_FOLDS_ARG=()
if [ -n "${MAX_FOLDS:-}" ]; then
  MAX_FOLDS_ARG=(--max-folds "$MAX_FOLDS")
fi
REUSE_ARG=()
if [ "$REUSE_CHECKPOINTS" -eq 1 ]; then
  REUSE_ARG=(--reuse-checkpoints)
fi

python v-jepa/finetune_sails_vjepa2_cv.py \
  --csv-dir "$CSV_DIR" \
  --clips-root "$CLIPS_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --model-id "$MODEL_ID" \
  --batch-size "${BATCH_SIZE:-1}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --num-epochs "${NUM_EPOCHS:-10}" \
  --lr "${LR:-1e-5}" \
  --accumulation-steps "${ACCUMULATION_STEPS:-4}" \
  --log-interval "${LOG_INTERVAL:-50}" \
  --frames-per-clip "${FRAMES_PER_CLIP:-16}" \
  --topk "${TOPK:-2}" \
  --start-fold "$START_FOLD" \
  --wandb-mode "$WANDB_MODE" \
  --wandb-project "$WANDB_PROJECT" \
  --run-prefix "$RUN_PREFIX" \
  --log-level "$LOG_LEVEL" \
  ${EXTRA_ARGS:-} \
  "${MAX_FOLDS_ARG[@]}" \
  "${REUSE_ARG[@]}"
