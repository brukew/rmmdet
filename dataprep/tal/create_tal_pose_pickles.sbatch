#!/bin/bash
#SBATCH --job-name=tal_pose_pkl
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=slurm-logs/create_tal_pose_pickles_%j.out
#SBATCH --error=slurm-logs/create_tal_pose_pickles_%j.err

# ============================================================================
# Create pyskl annotation pickle files for TAL windows
# Generates pose data for PoseC3D/ST-GCN training with background class
#
# Submit from the repo root:  sbatch dataprep/tal/create_tal_pose_pickles.sbatch
#
# Processes all 4 configurations:
#   - CV 4-class (4 folds, 5 classes with background)
#   - CV 5-class (4 folds, 6 classes with background)
#   - Single 4-class (train/val/test, 5 classes with background)
#   - Single 5-class (train/val/test, 6 classes with background)
#
# Writes ~149 MB TAL window pickles under pyskl/data/sails/tal/ (git-ignored;
# also available in checkpoints_root). Only needed if you change TAL splits.
# ============================================================================

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
source ~/.bashrc
cd "${REPO_ROOT}"
conda activate dataprep

# Disable Python output buffering so logs appear in real-time
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "=========================================="

# Paths
SCRIPT="${REPO_ROOT}/pyskl/tools/data/create_sails_annotations.py"
TAL_SPLITS_BASE="${REPO_ROOT}/dataprep/tal"
OUTPUT_BASE="${REPO_ROOT}/pyskl/data/sails/tal"

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
