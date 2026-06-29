#!/usr/bin/env bash
# SLURM job: convert finished two-stage predictions into OpenTAD-native outputs.
# Usage:
#   sbatch --export=FOLDS=1,2 two-stg/run_opentad_postprocess.sh

#SBATCH --job-name=two_stg_otad
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/orcd/data/satra/001/users/brukew/actreg/two-stg/logs/two_stage_opentad_%j.out
#SBATCH --error=/orcd/data/satra/001/users/brukew/actreg/two-stg/logs/two_stage_opentad_%j.err

set -euo pipefail

FOLDS="${FOLDS:?FOLDS env var required, e.g. 1,2 or 0}"
SOURCE_JOB_IDS="${SOURCE_JOB_IDS:-}"

echo "Two-Stage OpenTAD Postprocess"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  $(date)"
echo "Folds: $FOLDS"

mkdir -p /orcd/data/satra/001/users/brukew/actreg/two-stg/logs

if [[ -f ~/.bashrc ]]; then
  set +u
  source ~/.bashrc
  set -u
fi

cd /orcd/data/satra/001/users/brukew/actreg
set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opentad
set -u

OUTPUT_ROOT="/orcd/data/satra/001/users/brukew/actreg/two-stg/eval_results"

IFS=',' read -r -a fold_array <<< "$FOLDS"
for fold in "${fold_array[@]}"; do
  pred_csv="${OUTPUT_ROOT}/fold${fold}/predictions.csv"
  out_dir="${OUTPUT_ROOT}/fold${fold}"

  if [[ ! -f "$pred_csv" ]]; then
    echo "Missing predictions for fold ${fold}: ${pred_csv}"
    exit 1
  fi

  echo "Running OpenTAD eval for fold ${fold}"
  python two-stg/opentad_eval.py \
    --fold "$fold" \
    --predictions-csv "$pred_csv" \
    --output-dir "$out_dir"

  python two-stg/write_info_json.py \
    --fold "$fold" \
    --output-dir "$out_dir" \
    --source-job-ids "$SOURCE_JOB_IDS" \
    --postprocess-job-id "${SLURM_JOB_ID:-}" \
    --postprocess-dependency "${SLURM_JOB_DEPENDENCY:-}"
done

echo "OpenTAD postprocess done. $(date)"
