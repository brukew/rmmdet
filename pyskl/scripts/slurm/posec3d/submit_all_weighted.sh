#!/bin/bash
# Submit all PoseC3D weighted training jobs

echo "Submitting PoseC3D weighted training jobs..."

cd /orcd/data/satra/001/users/brukew/actreg/pyskl/scripts/slurm/posec3d

# Single splits
echo "Submitting 4-class single weighted..."
sbatch train_4class_single_weighted.sh

echo "Submitting 5-class single weighted..."
sbatch train_5class_single_weighted.sh

# CV splits
echo "Submitting 4-class CV weighted..."
sbatch train_4class_cv_weighted.sh

echo "Submitting 5-class CV weighted..."
sbatch train_5class_cv_weighted.sh

echo ""
echo "All PoseC3D weighted jobs submitted! Check status with: squeue -u \$USER"

