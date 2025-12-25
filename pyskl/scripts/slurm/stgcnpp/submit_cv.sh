#!/bin/bash
# Submit cv STGCN++ 4-stream training jobs for SAILS
# Usage: bash scripts/slurm/stgcnpp/submit_cv.sh [--dry-run]

cd /orcd/data/satra/001/users/brukew/actreg/pyskl

DRY_RUN=false
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE ==="
fi

# Create log directory
mkdir -p /orcd/data/satra/001/users/brukew/pyskl_logs/stgcnpp

echo "=========================================="
echo "STGCN++ SAILS 4-Stream CV Training"
echo "Date: $(date)"
echo "=========================================="
echo ""
echo "Each job trains all 4 modalities (j, b, jm, bm) and computes fusion."
echo "Fusion weights: 2*j + 2*b + 1*jm + 1*bm"
echo ""

SCRIPTS=(
    "scripts/slurm/stgcnpp/train_4class_cv_4stream.sh"
    "scripts/slurm/stgcnpp/train_5class_cv_4stream.sh"
)

JOB_IDS=()

for SCRIPT in "${SCRIPTS[@]}"; do
    echo "Submitting: $SCRIPT"
    
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] Would run: sbatch $SCRIPT"
    else
        JOB_ID=$(sbatch $SCRIPT | awk '{print $4}')
        JOB_IDS+=($JOB_ID)
        echo "  Submitted job: $JOB_ID"
    fi
done

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Total jobs: ${#JOB_IDS[@]}"
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Training per job:"
echo "  - CV (3-fold):  4 modalities × 3 folds × 24 epochs"
echo ""
echo "Monitor with: squeue -u $USER"
echo "Logs at: /orcd/data/satra/001/users/brukew/pyskl_logs/stgcnpp/"
echo "=========================================="
