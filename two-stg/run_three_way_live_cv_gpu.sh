#!/usr/bin/env bash
# Submit all 3 folds: full 3-way pipeline with live proposal skeleton scores + decontaminated MLP.
#
# Each job is long (pickle + 5 skeleton runs + V-JEPA2 eval). If pickles already
# exist under two-stg/proposal_pickles/, export SKIP_BUILD_ONLY=1 to skip rebuild.
#
# Usage (from actreg root):
#   SKIP_BUILD_ONLY=1 bash two-stg/run_three_way_live_cv_gpu.sh
#   bash two-stg/run_three_way_live_cv_gpu.sh   # rebuilds pickles each job (slow)

set -euo pipefail

cd /orcd/data/satra/001/users/brukew/actreg

for fold in 0 1 2; do
  BUNDLE="${PWD}/two-stg/fusion_checkpoints/three_way/fold_${fold}"
  if [[ ! -f "${BUNDLE}/mlp_state.pt" ]]; then
    echo "ERROR: Missing ${BUNDLE}/mlp_state.pt — run fusion/export_three_way_deploy_checkpoint.py --all-folds"
    exit 1
  fi
  echo "Submitting 3-way LIVE fold ${fold}..."
  if [[ "${SKIP_BUILD_ONLY:-0}" == "1" ]]; then
    sbatch --export=FOLD="${fold}",SKIP_BUILD_ONLY=1 two-stg/run_three_way_live_fold_gpu.sh
  else
    sbatch --export=FOLD="${fold}" two-stg/run_three_way_live_fold_gpu.sh
  fi
done

echo "Submitted 3 jobs. Monitor: squeue -u \$USER"
