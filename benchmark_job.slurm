#!/bin/bash
#SBATCH --job-name=breaching_benchmark
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
# Load modules or activate conda environment here

echo "Job started on $(hostname) at $(date)"
echo "Arguments: $@"

# -----------------------------------------------------------------------------
# Execution
# -----------------------------------------------------------------------------
# Run the command passed as arguments to the script
# Example usage: sbatch benchmark_job.slurm python benchmark_breaches.py ...

mkdir -p logs
srun "$@"

echo "Job finished at $(date)"
