#!/usr/bin/env bash
# SLURM job: run two-stage TAL evaluation (binary ActionFormer + V-JEPA2 classification) on a GPU node.
# Usage: sbatch two-stg/run_two_stage_cv_gpu.sh
# (submit from actreg repo root, or use absolute path to script)

#SBATCH --job-name=two_stg_tal
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3:20:00
#SBATCH --output=/orcd/data/satra/001/users/brukew/actreg/two-stg/logs/two_stage_cv_%j.out
#SBATCH --error=/orcd/data/satra/001/users/brukew/actreg/two-stg/logs/two_stage_cv_%j.err

set -e

echo "=============================================="
echo "Two-Stage TAL Eval (Binary AF + V-JEPA2 4-class)"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  $(date)"
echo "=============================================="

mkdir -p /orcd/data/satra/001/users/brukew/actreg/two-stg/logs
export PYTHONUNBUFFERED=1

if [[ -f ~/.bashrc ]]; then
  source ~/.bashrc
fi

cd /orcd/data/satra/001/users/brukew/actreg
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vjepa2

bash two-stg/run_two_stage_cv.sh

echo "Done. $(date)"
