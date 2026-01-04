#!/bin/bash
#SBATCH --job-name=tal_pkl_4c
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/orcd/data/satra/001/users/brukew/actreg/dataprep/tal/logs/create_tal_pose_pickles_4class_%j.out
#SBATCH --error=/orcd/data/satra/001/users/brukew/actreg/dataprep/tal/logs/create_tal_pose_pickles_4class_%j.err

# ============================================================================
# Create pyskl annotation pickle files for TAL windows - CV 4-class ONLY
# Completes fold2 (fold0 and fold1 already exist)
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
SCRIPT="/orcd/data/satra/001/users/brukew/actreg/pyskl/tools/data/create_sails_annotations.py"
TAL_SPLITS_BASE="/orcd/data/satra/001/users/brukew/actreg/dataprep/tal"
OUTPUT_BASE="/orcd/data/satra/001/users/brukew/actreg/pyskl/data/sails/tal"

# Configuration
MIN_CONF=0.4

echo ""
echo "Script: ${SCRIPT}"
echo "Min keypoint confidence: ${MIN_CONF}"
echo "Output base: ${OUTPUT_BASE}"
echo ""

# ============================================================================
# CV 4-class TAL windows (5 classes with background)
# ============================================================================
echo "============================================================"
echo "Processing CV 4-class TAL windows (→ 5 classes with background)"
echo "============================================================"

python "${SCRIPT}" \
    --mode cv \
    --splits-dir "${TAL_SPLITS_BASE}/splits_cv_4class" \
    --output-dir "${OUTPUT_BASE}/cv_4class" \
    --windows \
    --num-classes 4 \
    --min-keypoint-conf ${MIN_CONF}

echo ""
echo "CV 4-class complete!"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="

echo ""
echo "Output files:"
find "${OUTPUT_BASE}/cv_4class" -name "*.pkl" -type f 2>/dev/null | sort

echo ""
echo "Pickle sizes:"
du -sh "${OUTPUT_BASE}/cv_4class"/*/ 2>/dev/null || echo "(no output dirs yet)"

