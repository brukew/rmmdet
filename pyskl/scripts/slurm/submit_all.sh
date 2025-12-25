#!/bin/bash
#
# Submit all 4 PoseC3D training jobs to SLURM
#
# Usage:
#   bash scripts/slurm/submit_all.sh
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "Submitting PoseC3D Training Jobs"
echo "=============================================="
echo ""

# Submit each job
echo "1. 4-class single split..."
JOB1=$(sbatch --parsable $SCRIPT_DIR/train_4class_single.sh)
echo "   Job ID: $JOB1"

echo "2. 4-class CV (3 folds)..."
JOB2=$(sbatch --parsable $SCRIPT_DIR/train_4class_cv.sh)
echo "   Job ID: $JOB2"

echo "3. 5-class single split..."
JOB3=$(sbatch --parsable $SCRIPT_DIR/train_5class_single.sh)
echo "   Job ID: $JOB3"

echo "4. 5-class CV (3 folds)..."
JOB4=$(sbatch --parsable $SCRIPT_DIR/train_5class_cv.sh)
echo "   Job ID: $JOB4"

echo ""
echo "=============================================="
echo "All jobs submitted!"
echo "=============================================="
echo ""
echo "Monitor with:"
echo "  squeue -u $USER"
echo ""
echo "Logs will be saved to:"
echo "  /orcd/data/satra/001/users/brukew/pyskl_logs/posec3d/"
echo ""
echo "Results will be in:"
echo "  work_dirs/posec3d/single/4class_conf04/"
echo "  work_dirs/posec3d/cv/4class_conf04/"
echo "  work_dirs/posec3d/single/5class_conf04/"
echo "  work_dirs/posec3d/cv/5class_conf04/"

