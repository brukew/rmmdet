#!/bin/bash -l
# Re-run V-JEPA2 evaluation on existing 4-class checkpoints to regenerate CSVs with full probability scores.
# This is needed for late fusion with PoseC3D.
#
# Usage:
#   sbatch rerun_vjepa_eval_4class.sh
#
# Logs: /orcd/data/satra/001/users/brukew/vjepa_rmm_logs

#SBATCH -J vjepa_reeval
#SBATCH -p mit_preemptable
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH -t 4:00:00
#SBATCH --requeue
#SBATCH -o /orcd/data/satra/001/users/brukew/vjepa_rmm_logs/vjepa_reeval_4class_%j.out
#SBATCH -e /orcd/data/satra/001/users/brukew/vjepa_rmm_logs/vjepa_reeval_4class_%j.err

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
# Paths - same hyperparams as existing 4-class crop run
# ============================================================================
CSV_DIR=/orcd/data/satra/001/users/brukew/actreg/dataprep/splits/cv_splits_4class
CLIPS_ROOT="$CLASSIFICATION_CLIPS"
OUTPUT_ROOT=/orcd/data/satra/001/users/brukew/actreg/v-jepa/runs/vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop_4cls
MODEL_ID=facebook/vjepa2-vitl-fpc16-256-ssv2

# Cropping paths
MASK_CACHE_BASE="$CACHE_FOR_TRACKING"
SAM3_PARSED_CSV=/orcd/data/satra/001/users/brukew/actreg/dataprep/rmm_sam3_parsed.csv
VIDEO_META_JSON=/orcd/data/satra/001/users/brukew/actreg/dataprep/video_meta.json

echo "============================================================"
echo "V-JEPA2 Re-Evaluation (4-class CV with cropping)"
echo "============================================================"
echo "Output: $OUTPUT_ROOT"
echo "Purpose: Regenerate CSVs with per-class probability scores for late fusion"
echo "============================================================"

python actreg/v-jepa/finetune_sails_vjepa2_cv_crop.py \
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

