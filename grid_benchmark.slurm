#!/bin/bash
#SBATCH --job-name=bench_grid
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-8  # 9 combinations total (0 to 8)

# -----------------------------------------------------------------------------
# Grid Definition
# -----------------------------------------------------------------------------
# Define your hyperparameters here as bash arrays.
# All arrays must have the same length as the number of jobs (9 in this case).
# You can generate these parameter lists using a simple python script if the grid is large.

# Combination mapping:
# ID: 0 1 2 3 4 5 6 7 8
# LR: .1 .5 1 .1 .5 1 .1 .5 1
# TV: .01 .01 .01 .1 .1 .1 1 1 1

LRS=(0.1 0.5 1.0 0.1 0.5 1.0 0.1 0.5 1.0)
TVS=(0.01 0.01 0.01 0.1 0.1 0.1 1.0 1.0 1.0)

# Get current task parameters
CURRENT_LR=${LRS[$SLURM_ARRAY_TASK_ID]}
CURRENT_TV=${TVS[$SLURM_ARRAY_TASK_ID]}

echo "Running Job Array ID: $SLURM_ARRAY_TASK_ID"
echo "Parameters: LR=$CURRENT_LR, TV_SCALE=$CURRENT_TV"

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
# module load cuda/11.3
# source activate breaching

mkdir -p logs

# -----------------------------------------------------------------------------
# Execution
# -----------------------------------------------------------------------------
python benchmark_breaches.py \
    case=10_fed \
    attack=modern \
    attack.optim.step_size=$CURRENT_LR \
    attack.regularization.total_variation.scale=$CURRENT_TV \
    case.data.partition=unique-users \
    dryrun=True # Set to False for actual runs
