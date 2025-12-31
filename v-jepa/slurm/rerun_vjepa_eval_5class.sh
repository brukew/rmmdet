#!/bin/bash -l
# Re-run V-JEPA2 evaluation on existing 5-class checkpoints to regenerate CSVs with full probability scores.
# This is needed for late fusion with PoseC3D/STGCN++.
#
# Usage:
#   sbatch rerun_vjepa_eval_5class.sh
#
# Models being fused (5-class):
#   - V-JEPA2 5-class crop: f64_lr1e-5_bs1_acc8_ep20_crop (this script regenerates predictions)
#   - PoseC3D 5-class weighted: pyskl/work_dirs/posec3d/cv/5class_conf04_weighted/ (best PoseC3D)
#   - STGCN++ 5-class non-weighted 4-stream: pyskl/work_dirs/stgcnpp/cv/5class_conf04_4stream/
#
# Logs: /orcd/data/satra/001/users/brukew/vjepa_rmm_logs

#SBATCH -J vjepa_reeval_5cls
#SBATCH -p pi_satra
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH -t 4:00:00
#SBATCH -o /orcd/data/satra/001/users/brukew/vjepa_rmm_logs/vjepa_reeval_5class_%j.out
#SBATCH -e /orcd/data/satra/001/users/brukew/vjepa_rmm_logs/vjepa_reeval_5class_%j.err

set -eo pipefail

if [ -f ~/.bashrc ]; then
  source ~/.bashrc
fi

cd /orcd/data/satra/001/users/brukew
conda activate vjepa2

LOG_DIR=/orcd/data/satra/001/users/brukew/vjepa_rmm_logs
mkdir -p "$LOG_DIR"

# ============================================================================
# Paths - same hyperparams as existing 5-class crop run
# ============================================================================
CSV_DIR=/orcd/data/satra/001/users/brukew/actreg/dataprep/splits/cv_splits
CLIPS_ROOT=/orcd/scratch/bcs/001/sensein/sails/rmm/classification_clips
OUTPUT_ROOT=/orcd/data/satra/001/users/brukew/actreg/v-jepa/runs/vjepa2_rmm_cv/f64_lr1e-5_bs1_acc8_ep20_crop
MODEL_ID=facebook/vjepa2-vitl-fpc16-256-ssv2

# Cropping paths
MASK_CACHE_BASE=/orcd/scratch/bcs/001/sensein/sails/cache_for_tracking
SAM3_PARSED_CSV=/orcd/data/satra/001/users/brukew/actreg/dataprep/rmm_sam3_parsed.csv
VIDEO_META_JSON=/orcd/data/satra/001/users/brukew/actreg/dataprep/video_meta.json

echo "============================================================"
echo "V-JEPA2 Re-Evaluation (5-class CV with cropping)"
echo "============================================================"
echo "Output: $OUTPUT_ROOT"
echo "CSV splits: $CSV_DIR"
echo "Purpose: Regenerate CSVs with per-class probability scores for late fusion"
echo ""
echo "After this completes, run fusion with:"
echo "  - PoseC3D 5-class weighted: pyskl/work_dirs/posec3d/cv/5class_conf04_weighted/"
echo "  - STGCN++ 5-class 4-stream: pyskl/work_dirs/stgcnpp/cv/5class_conf04_4stream/"
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
echo ""
echo "Next steps - run 5-class fusion:"
echo "  sbatch actreg/fusion/slurm/run_fusion_5class_cv.sh"
echo "============================================================"


