#!/bin/bash

# -----------------------------------------------------------------------------

# Master Grid Search Launcher (2-GPU Parallel Version)

# -----------------------------------------------------------------------------

#SBATCH --job-name=gias_grid_master
#SBATCH --output=slurm_logs/master_%j.out
#SBATCH --error=slurm_logs/master_%j.err
#SBATCH --partition=ada
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

mkdir -p slurm_logs

echo "Activating virtual environment..."

if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
echo "ERROR: SLURM_SUBMIT_DIR is not set"
exit 1
fi

echo "SLURM_SUBMIT_DIR = ${SLURM_SUBMIT_DIR}"

BASE_DIR="${SLURM_SUBMIT_DIR}"
PROJECT_ROOT="$(cd "${BASE_DIR}/.." && pwd)"

echo "BASE_DIR     = ${BASE_DIR}"
echo "PROJECT_ROOT = ${PROJECT_ROOT}"

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/venv/bin/activate"

echo "Python path: $(which python)"
python --version

cd "${BASE_DIR}"

echo "Virtual environment activated successfully."

# -----------------------------------------------------------------------------

# Grid definition

# -----------------------------------------------------------------------------

ATTACK=gias

BATCH_SIZES=(1 1 1 1)
PP_SCALES=(0.01 0.001 0.0001 0.00001)                # regularization.patch_prior.scale
STEP_SIZES=(0.01)
W_STEP_SIZES=(0.001)
OBJ_SCALES=(1.0)
TV_SCALES=(1e-3)

# -----------------------------------------------------------------------------

# GPU scheduling

# -----------------------------------------------------------------------------

MAX_GPU=2
GPU_ID=0
count=0

# -----------------------------------------------------------------------------

# Grid loop

# -----------------------------------------------------------------------------

for bs in "${BATCH_SIZES[@]}"; do
for ss in "${STEP_SIZES[@]}"; do
for wss in "${W_STEP_SIZES[@]}"; do
for os in "${OBJ_SCALES[@]}"; do
for tv in "${TV_SCALES[@]}"; do
for pp in "${PP_SCALES[@]}"; do
JOB_NAME="${ATTACK}_bs${bs}_ss${ss}_wss${wss}_os${os}_tv${tv}_pp${pp}_count${count}"

echo "Launching job $count on GPU $GPU_ID"

CUDA_VISIBLE_DEVICES=${GPU_ID} python benchmark_breaches.py \
    case/data=CIFAR10 \
    case.model=vit_small \
    case.server.pretrained=True \
    save_reconstruction=true \
    case=1_single_image_small \
    case.data.batch_size=${bs} \
    case.user.num_data_points=${bs} \
    case.peft.type=lora \
    case.peft.lora.rank=128 \
    case.peft.lora.alpha=256 \
    attack=${ATTACK} \
    dryrun=False \
    attack.restarts.num_trials=3 \
    num_trials=3 \
    attack.objective.type=cosine-similarity \
    attack.objective.posembed_trainable=False \
    attack.optim.step_size=${ss} \
    attack.optim_w.step_size=${wss} \
    attack.objective.scale=${os} \
    attack.regularization.total_variation.scale=${tv} \
    attack.regularization.patch_prior.scale=${pp} \
    attack.regularization.patch_prior.patch_size=16 \
    job_name="${JOB_NAME}" &

GPU_ID=$((GPU_ID + 1))
count=$((count + 1))

if [ ${GPU_ID} -eq ${MAX_GPU} ]; then
    wait
    GPU_ID=0
fi

done
done
done
done
done
done

wait

echo "Grid search completed."
