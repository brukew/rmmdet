#!/bin/bash
#SBATCH --job-name=tal_pose_pkl
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/orcd/data/satra/001/users/brukew/actreg/dataprep/tal/logs/create_tal_pose_pickles_%j.out
#SBATCH --error=/orcd/data/satra/001/users/brukew/actreg/dataprep/tal/logs/create_tal_pose_pickles_%j.err

# ============================================================================
# Create pyskl annotation pickle files for TAL windows
# Generates pose data for PoseC3D/ST-GCN training with background class
#
# Processes all 4 configurations:
#   - CV 4-class (4 folds, 5 classes with background)
#   - CV 5-class (4 folds, 6 classes with background)
#   - Single 4-class (train/val/test, 5 classes with background)
#   - Single 5-class (train/val/test, 6 classes with background)
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
# 1. CV 4-class TAL windows (5 classes with background)
# ============================================================================
echo "============================================================"
echo "[1/4] Processing CV 4-class TAL windows (→ 5 classes with background)"
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
# 2. CV 5-class TAL windows (6 classes with background)
# ============================================================================
echo "============================================================"
echo "[2/4] Processing CV 5-class TAL windows (→ 6 classes with background)"
echo "============================================================"

python "${SCRIPT}" \
    --mode cv \
    --splits-dir "${TAL_SPLITS_BASE}/splits_cv_5class" \
    --output-dir "${OUTPUT_BASE}/cv_5class" \
    --windows \
    --num-classes 5 \
    --min-keypoint-conf ${MIN_CONF}

echo ""
echo "CV 5-class complete!"
echo ""

# ============================================================================
# 3. Single 4-class TAL windows (5 classes with background)
# ============================================================================
echo "============================================================"
echo "[3/4] Processing Single 4-class TAL windows (→ 5 classes with background)"
echo "============================================================"

python "${SCRIPT}" \
    --mode single \
    --splits-dir "${TAL_SPLITS_BASE}/splits_single_4class" \
    --output-dir "${OUTPUT_BASE}/single_4class" \
    --windows \
    --num-classes 4 \
    --min-keypoint-conf ${MIN_CONF}

echo ""
echo "Single 4-class complete!"
echo ""

# ============================================================================
# 4. Single 5-class TAL windows (6 classes with background)
# ============================================================================
echo "============================================================"
echo "[4/4] Processing Single 5-class TAL windows (→ 6 classes with background)"
echo "============================================================"

python "${SCRIPT}" \
    --mode single \
    --splits-dir "${TAL_SPLITS_BASE}/splits_single_5class" \
    --output-dir "${OUTPUT_BASE}/single_5class" \
    --windows \
    --num-classes 5 \
    --min-keypoint-conf ${MIN_CONF}

echo ""
echo "Single 5-class complete!"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="

echo ""
echo "Output files:"
find "${OUTPUT_BASE}" -name "*.pkl" -type f 2>/dev/null | sort

echo ""
echo "Pickle sizes:"
du -sh "${OUTPUT_BASE}"/*/ 2>/dev/null || echo "(no output dirs yet)"
