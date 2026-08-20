#!/bin/bash -l
# Re-run V-JEPA2 evaluation on existing 4-class checkpoints to regenerate CSVs with full probability scores.
# This is needed for late fusion with PoseC3D.
#
# Usage:
#   sbatch rerun_vjepa_eval_4class.sh
#
# Logs: ${REPO_ROOT}/vjepa_rmm_logs

#SBATCH -J vjepa_reeval
#SBATCH -p mit_preemptable
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH -t 4:00:00
#SBATCH --requeue
#SBATCH -o slurm-logs/vjepa_reeval_4class_%j.out
#SBATCH -e slurm-logs/vjepa_reeval_4class_%j.err

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
# Paths - same hyperparams as existing 4-class crop run
# ============================================================================
CSV_DIR=${REPO_ROOT}/dataprep/splits/cv_splits_4class
CLIPS_ROOT="$CLASSIFICATION_CLIPS"
OUTPUT_ROOT=${REPO_ROOT}/v-jepa/runs/vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop_4cls
MODEL_ID=facebook/vjepa2-vitl-fpc16-256-ssv2

# Cropping paths
MASK_CACHE_BASE="$CACHE_FOR_TRACKING"
SAM3_PARSED_CSV=${REPO_ROOT}/dataprep/rmm_sam3_parsed.csv
VIDEO_META_JSON=${REPO_ROOT}/dataprep/video_meta.json

echo "============================================================"
echo "V-JEPA2 Re-Evaluation (4-class CV with cropping)"
echo "============================================================"
echo "Output: $OUTPUT_ROOT"
echo "Purpose: Regenerate CSVs with per-class probability scores for late fusion"
echo "============================================================"

python v-jepa/finetune_sails_vjepa2_cv_crop.py \
  --csv-dir "$CSV_DIR" \
  --clips-root "$CLIPS_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --model-id "$MODEL_ID" \
  --batch-size 1 \
  --num-workers 8 \
  --num-epochs 20 \
  --lr 1e-5 \
  --accumulation-steps 8 \
  --frames-per-clip 64 \
  --topk 2 \
  --wandb-mode disabled \
  --log-level INFO \
  --enable-crop \
  --mask-cache-base "$MASK_CACHE_BASE" \
  --sam3-parsed-csv "$SAM3_PARSED_CSV" \
  --video-meta-json "$VIDEO_META_JSON" \
  --reuse-checkpoints

echo "============================================================"
echo "Done! CSVs regenerated with per-class scores at: $OUTPUT_ROOT"
echo "============================================================"

