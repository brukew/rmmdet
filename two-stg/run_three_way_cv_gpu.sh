#!/usr/bin/env bash
# Submit all 3 folds for two-stage TAL with 3-way fusion Stage-2 (GPU jobs).
# Usage (from actreg root):
#   bash two-stg/run_three_way_cv_gpu.sh
#
# Prerequisite:
#   python fusion/export_three_way_deploy_checkpoint.py --all-folds

set -e

cd /orcd/data/satra/001/users/brukew/actreg

echo "Submitting 3-way fusion two-stage TAL jobs (one GPU job per fold)..."

for fold in 0 1 2; do
  BUNDLE="${PWD}/two-stg/fusion_checkpoints/three_way/fold_${fold}"
  if [[ ! -f "${BUNDLE}/mlp_state.pt" ]]; then
    echo "ERROR: Missing ${BUNDLE}/mlp_state.pt — run export script first."
    exit 1
  fi
  echo "Submitting fold ${fold}..."
  sbatch --export=FOLD="${fold}" two-stg/run_three_way_fold_gpu.sh
done

echo ""
echo "All 3 folds submitted. Monitor: squeue -u \$USER"
echo "Logs: two-stg/logs/two_stage_3way_fold*.out"
echo "Results: two-stg/eval_results_3way/fold{0,1,2}/"
