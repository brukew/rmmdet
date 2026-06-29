#!/usr/bin/env bash
# Submit all 3 folds for 5-class two-stage TAL evaluation.
# Usage: bash two-stg/run_5class_cv_gpu.sh

set -e

cd /orcd/data/satra/001/users/brukew/actreg

echo "Submitting 5-class two-stage TAL jobs for all folds..."

for fold in 0 1 2; do
    echo "Submitting fold ${fold}..."
    sbatch --export=FOLD=${fold} two-stg/run_5class_fold_gpu.sh
done

echo ""
echo "All 3 folds submitted. Monitor with: squeue -u \$USER"
echo "Results will be in: two-stg/eval_results_5class/fold{0,1,2}/"
