#!/bin/bash
# -----------------------------------------------------------------------------
# Master Grid Search Launcher
# -----------------------------------------------------------------------------
# This script defines a hyperparameter grid and submits individual Slurm jobs
# for each combination.
#
# Usage: sbatch run_grid.slurm
# -----------------------------------------------------------------------------

#SBATCH --job-name=grid_master
#SBATCH --output=logs/master_%j.out
#SBATCH --error=logs/master_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=2G
#SBATCH --partition=debug # Use a low-priority partition for the launcher if available

# 1. Define the grid
STEP_SIZES=(0.1 0.5 1.0)
TV_SCALES=(0.01 0.1 1.0)
PATCH_SCALES=(0.1 1.0)

# 2. Iterate through combinations
count=0
for ss in "${STEP_SIZES[@]}"; do
    for tv in "${TV_SCALES[@]}"; do
        for ps in "${PATCH_SCALES[@]}"; do
            
            JOB_NAME="gradvit_ss${ss}_tv${tv}_ps${ps}"
            
            echo "Submitting job $count: $JOB_NAME"
            
            # Submit the actual benchmark job
            # We use --wrap to send the command directly to sbatch
            sbatch --job-name="$JOB_NAME" \
                   --output="logs/%x_%j.out" \
                   --error="logs/%x_%j.err" \
                   --time=04:00:00 \
                   --mem=16G \
                   --gres=gpu:1 \
                   --wrap="python benchmark_breaches.py \
                            case=1_single_image_small \
                            attack=gradvit \
                            attack.optim.step_size=$ss \
                            attack.regularization.total_variation.scale=$tv \
                            attack.regularization.patch_prior.scale=$ps \
                            dryrun=True"
            
            count=$((count + 1))
        done
    done
done

echo "Submitted $count jobs in total."
