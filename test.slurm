#!/bin/bash
# -----------------------------------------------------------------------------
# Master Grid Search Launcher (Robust Version)
# -----------------------------------------------------------------------------
# Submits GPU jobs for each hyperparameter combination.
# This job itself runs on a low-priority CPU partition.
#
# Usage:
#   sbatch run_grid.slurm
# -----------------------------------------------------------------------------

#SBATCH --job-name=testing
#SBATCH --output=slurm_logs/master_%j.out
#SBATCH --error=slurm_logs/master_%j.err
#SBATCH --partition=short
#SBATCH --nodelist=cn5
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# echo "Grid master started on $(hostname) at $(date)"
# echo "Working directory: $(pwd)"

# Load Python module
module avail cuda
module load cuda/12.4
which nvcc
nvcc --version
# export TORCH_CUDA_ARCH_LIST=8.9

mkdir -p slurm_logs
# -----------------------------------------------------------------------------
# Activate virtual environment (SLURM-safe)
# -----------------------------------------------------------------------------

echo "Activating virtual environment..."

if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
    echo "ERROR: SLURM_SUBMIT_DIR is not set"
    exit 1
fi

echo "SLURM_SUBMIT_DIR = ${SLURM_SUBMIT_DIR}"

# breaching/ is the submit directory
BASE_DIR="${SLURM_SUBMIT_DIR}"

# project root is one level above
PROJECT_ROOT="$(cd "${BASE_DIR}/.." && pwd)"

echo "BASE_DIR     = ${BASE_DIR}"
echo "PROJECT_ROOT = ${PROJECT_ROOT}"

cd "${PROJECT_ROOT}"

if [[ ! -f bin/activate ]]; then
    echo "ERROR: Virtual environment not found at ${PROJECT_ROOT}/bin/activate"
    exit 1
fi

source bin/activate

# Sanity checks
which python
python --version

# Return to breaching
cd "${BASE_DIR}"

echo "Virtual environment activated successfully."


ATTACK=gismn
# -----------------------------------------------------------------------------
# Single run (no loops)
# -----------------------------------------------------------------------------
BS=1
SS=0.1
OS=1.0
# TV=0.01
PE_SCALE=0.1

JOB_NAME="${ATTACK}_bs${BS}_ss${SS}_os${OS}_tv${TV}_pes${PE_SCALE}"

echo "Running single job: $JOB_NAME"

python -c "import importlib.util; print(importlib.util.find_spec('distutils'))"

python -c "import breaching.attacks.auxiliaries.stylegan_xl.dnnlib.util as u; import inspect; print(u.__file__); import pathlib; print('try/except strtobool' in pathlib.Path(u.__file__).read_text())"

python -c "import pathlib; p=pathlib.Path('breaching/attacks/auxiliaries/stylegan_xl/dnnlib/util.py'); print(p.resolve()); print(p.read_text().splitlines()[31])"

python -c "import torch; print(torch.version.cuda)"

ls -l breaching/attacks/auxiliaries/stylegan_xl/dnnlib/util.py

python benchmark_breaches.py \
    case=1_single_image_small \
    case/data=CIFAR10 \
    case.model=vit_custom \
    case.server.pretrained=False \
    case.data.batch_size=${BS} \
    case.user.num_data_points='${case.data.batch_size}' \
    attack=${ATTACK} \
    attack.optim.step_size=${SS} \
    attack.objective.scale=${OS} \
    attack.objective.posembed_scale=${PE_SCALE} \
    save_reconstruction=true \
    dryrun=False \
    job_name="${JOB_NAME}" \
    num_trials=3 \
    case.impl.threads=4

echo "Single job finished at $(date)"

# attack.regularization.total_variation.scale=${TV} \
