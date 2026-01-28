#!/bin/bash
# -----------------------------------------------------------------------------
# Master Grid Search Launcher (Robust Version)
# -----------------------------------------------------------------------------
# Submits GPU jobs for each hyperparameter combination.
# This job itself runs on a low-priority CPU partition.
#
# Usage:
#   mkdir -p slurm_logs
#   sbatch run_grid.slurm
# -----------------------------------------------------------------------------

#SBATCH --job-name=grid_master
#SBATCH --partition=long
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_logs/master_%j.out
#SBATCH --error=slurm_logs/master_%j.err

set -euo pipefail
set -x

echo "Grid master started on $(hostname) at $(date)"
echo "Working directory: $(pwd)"

# Load Python module
module load python3.10

# -----------------------------------------------------------------------------
# 1. Define the grid
# -----------------------------------------------------------------------------
STEP_SIZES=(0.1)
TV_SCALES=(0.01 0.1)
PATCH_SCALES=(0.1)

# -----------------------------------------------------------------------------
# 2. Iterate through combinations and submit jobs
# -----------------------------------------------------------------------------
count=0

for ss in "${STEP_SIZES[@]}"; do
    for tv in "${TV_SCALES[@]}"; do
        for ps in "${PATCH_SCALES[@]}"; do

            JOB_NAME="gradvit_ss${ss}_tv${tv}_ps${ps}"

            echo "Submitting job $count: $JOB_NAME"

            JOBID=$(sbatch \
                --job-name="$JOB_NAME" \
                --partition=long \
                --gres=gpu:1 \
                --time=04:00:00 \
                --mem=16G \
                --cpus-per-task=4 \
                --output="slurm_logs/%x_%j.out" \
                --error="slurm_logs/%x_%j.err" \
                --wrap="python benchmark_breaches.py \
                        case=1_single_image_small \
                        attack=gradvit \
                        attack.optim.step_size=${ss} \
                        attack.regularization.total_variation.scale=${tv} \
                        attack.regularization.patch_prior.scale=${ps} \
                        dryrun=True" \
                | awk '{print $4}')

            if [[ -z "${JOBID}" ]]; then
                echo "❌ Submission failed for $JOB_NAME"
                exit 1
            fi

            echo "✅ Submitted $JOB_NAME as JobID=${JOBID}"
            count=$((count + 1))

        done
    done
done

echo "Submitted $count jobs in total at $(date)"
echo "Grid master finished."