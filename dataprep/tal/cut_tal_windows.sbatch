#!/bin/bash
#SBATCH --job-name=cut_tal_windows
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/orcd/data/satra/001/users/brukew/actreg/dataprep/tal/logs/cut_tal_windows_%j.out
#SBATCH --error=/orcd/data/satra/001/users/brukew/actreg/dataprep/tal/logs/cut_tal_windows_%j.err

# ============================================================================
# Cut TAL window clips from full videos
# Uses create_clip_segments.py with window_id -> segment_id mapping
# ============================================================================

set -eo pipefail
source ~/.bashrc
cd /orcd/data/satra/001/users/brukew
conda activate dataprep

# Disable Python output buffering so logs appear in real-time
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "=========================================="

# Paths
SCRIPT_DIR="/orcd/data/satra/001/users/brukew/actreg/dataprep/clip_gen"
SPLITS_DIR="/orcd/data/satra/001/users/brukew/actreg/dataprep/tal/splits_cv_4class"
OUTPUT_DIR="/orcd/scratch/bcs/001/brukew/sails/tal_windows_4class"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "Input CSVs:"
echo "  - ${SPLITS_DIR}/fold_0_train_windows.csv"
echo "  - ${SPLITS_DIR}/fold_0_val_windows.csv"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Run clip generation
# - Using fold_0 train+val with dedup covers all unique windows
# - --no-include-end-second: TAL windows are already half-open [start, end)
# - --jobs 12: Use 12 parallel ffmpeg workers (leaving headroom for overhead)
# - --dedup-by-segment: Avoid re-encoding the same window_id across CSVs

cd "${SCRIPT_DIR}"

python create_clip_segments.py \
    --csv-files "${SPLITS_DIR}/fold_0_train_windows.csv" \
                "${SPLITS_DIR}/fold_0_val_windows.csv" \
    --output-dir "${OUTPUT_DIR}" \
    --no-include-end-second \
    --jobs 12 \
    --dedup-by-segment \
    --codec h264

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="

# Show summary
echo ""
echo "Output summary:"
ls -lh "${OUTPUT_DIR}" | head -20
echo ""
echo "Total clips generated:"
find "${OUTPUT_DIR}" -name "*.mp4" | wc -l

