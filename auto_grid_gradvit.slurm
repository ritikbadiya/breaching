#!/bin/bash
#SBATCH --job-name=grid_array_gradvit
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-299
#SBATCH --partition=long
#SBATCH --nodelist=cn4

# -----------------------------------------------------------------------------
# 1. Define the Grid
# -----------------------------------------------------------------------------
STEPS=(0.01 0.1 0.25 0.5 1.0)        # Length 5
TV_SCALES=(0.01 0.1 1.0)   # Length 3
PATCH_SCALES=(0.0001 0.001 0.01 0.1 1.0)     # Length 5
IMAGE_SCALE=(0.1 0.2 0.5 1.0) # Length 4

# Total length of the array should be: 5 * 3 * 5 * 4 = 300
# Run this once: sbatch --array=0-299 auto_grid_gradvit.slurm

# -----------------------------------------------------------------------------
# 2. Mathematical Mapping (Avoids typing out all 18 combinations)
# -----------------------------------------------------------------------------
# We use integer division (/) and modulo (%) to pick indices
# It's like converting a single number (Task ID) into 3D coordinates (Indices)

i=$SLURM_ARRAY_TASK_ID

# For 3 variables:
# index_z = i % len_z
# remainder = i / len_z
# index_y = remainder % len_y
# index_x = remainder / len_y

N_PATCH=${#PATCH_SCALES[@]}
N_TV=${#TV_SCALES[@]}
N_STEP=${#STEPS[@]}
N_IMAGE=${#IMAGE_SCALE[@]}

# Calculate Indices
idx_patch=$(( i % N_PATCH ))
remainder=$(( i / N_PATCH ))
idx_tv=$(( remainder % N_TV ))
idx_step=$(( remainder / N_TV ))
idx_image=$(( remainder / N_TV ))

# Get Values
CURRENT_STEP=${STEPS[$idx_step]}
CURRENT_TV=${TV_SCALES[$idx_tv]}
CURRENT_PATCH=${PATCH_SCALES[$idx_patch]}
CURRENT_IMAGE=${IMAGE_SCALE[$idx_image]}

echo "Task ID $i maps to: Step=$CURRENT_STEP, TV=$CURRENT_TV, Patch=$CURRENT_PATCH, Image=$CURRENT_IMAGE"

# -----------------------------------------------------------------------------
# 3. Execution
# -----------------------------------------------------------------------------
mkdir -p logs

python benchmark_breaches.py \
    case=1_single_image_small \
    attack=gradvit \
    attack.optim.step_size=$CURRENT_STEP \
    attack.regularization.total_variation.scale=$CURRENT_TV \
    attack.regularization.patch_prior.scale=$CURRENT_PATCH \
    attack.regularization.image_prior.scale=$CURRENT_IMAGE \
    dryrun=True
