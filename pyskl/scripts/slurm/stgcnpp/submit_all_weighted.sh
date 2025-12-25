#!/bin/bash
# Submit all STGCN++ weighted training jobs

echo "Submitting STGCN++ weighted 4-stream training jobs..."

cd /orcd/data/satra/001/users/brukew/actreg/pyskl/scripts/slurm/stgcnpp

# Single splits
echo "Submitting 4-class single weighted..."
sbatch train_4class_single_4stream_weighted.sh

echo "Submitting 5-class single weighted..."
sbatch train_5class_single_4stream_weighted.sh

# CV splits
echo "Submitting 4-class CV weighted..."
sbatch train_4class_cv_4stream_weighted.sh

echo "Submitting 5-class CV weighted..."
sbatch train_5class_cv_4stream_weighted.sh

echo ""
echo "All weighted jobs submitted! Check status with: squeue -u \$USER"

