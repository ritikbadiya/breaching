#!/bin/bash
#SBATCH --job-name=grid_chunked_gradvit
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-29  # 300 jobs / 10 cmds per job = 30 jobs total
#SBATCH --partition=long
#SBATCH --nodelist=cn4

# -----------------------------------------------------------------------------
# 1. Define the Grid
# -----------------------------------------------------------------------------
STEPS=(0.01 0.1 0.25 0.5 1.0)        # Length 5
TV_SCALES=(0.01 0.1 1.0)             # Length 3
PATCH_SCALES=(0.0001 0.001 0.01 0.1 1.0) # Length 5
IMAGE_SCALE=(0.1 0.2 0.5 1.0)        # Length 4

# Total combinations: 5 * 3 * 5 * 4 = 300
# Chunk size: How many runs to do per job?
CHUNK_SIZE=10

# -----------------------------------------------------------------------------
# 2. Chunk Loop
# -----------------------------------------------------------------------------
# Calculate the range of indices this job is responsible for
START_IDX=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
END_IDX=$(( START_IDX + CHUNK_SIZE - 1 ))

# Safety check for the last chunk
TOTAL_COMBS=300
if [ $END_IDX -ge $TOTAL_COMBS ]; then
    END_IDX=$(( TOTAL_COMBS - 1 ))
fi

echo "Job Array ID $SLURM_ARRAY_TASK_ID handling indices $START_IDX to $END_IDX"

mkdir -p logs

for (( i=START_IDX; i<=END_IDX; i++ )); do

    # -------------------------------------------------------------------------
    # 3. Mathematical Mapping
    # -------------------------------------------------------------------------
    
    # Grid dimensions
    N_PATCH=${#PATCH_SCALES[@]}
    N_TV=${#TV_SCALES[@]}
    N_STEP=${#STEPS[@]}
    N_IMAGE=${#IMAGE_SCALE[@]}
    
    # Calculate Indices from flat index 'i'
    # Order of unpacking matters: must match the intended nesting
    # Here we assume order: Image -> Step -> TV -> Patch (Inner most)
    
    # idx_patch = i % N_PATCH
    # i = i / N_PATCH
    # idx_tv = i % N_TV
    # i = i / N_TV 
    # ... and so on.
    
    # Temporary variable for math so we don't modify 'i'
    tmp_i=$i
    
    idx_image=$(( tmp_i % N_IMAGE ))
    tmp_i=$(( tmp_i / N_IMAGE ))
    
    idx_patch=$(( tmp_i % N_PATCH ))
    tmp_i=$(( tmp_i / N_PATCH ))
    
    idx_tv=$(( tmp_i % N_TV ))
    tmp_i=$(( tmp_i / N_TV ))
    
    idx_step=$(( tmp_i % N_STEP ))
    
    
    # Get Values
    CURRENT_STEP=${STEPS[$idx_step]}
    CURRENT_TV=${TV_SCALES[$idx_tv]}
    CURRENT_PATCH=${PATCH_SCALES[$idx_patch]}
    CURRENT_IMAGE=${IMAGE_SCALE[$idx_image]}
    
    echo "[run $i] Params: Step=$CURRENT_STEP, TV=$CURRENT_TV, Patch=$CURRENT_PATCH, Image=$CURRENT_IMAGE"
    
    # -------------------------------------------------------------------------
    # 4. Execution
    # -------------------------------------------------------------------------
    # Using specific log file for each run so they don't get mixed up in the main log
    param_tag="s${CURRENT_STEP}_tv${CURRENT_TV}_p${CURRENT_PATCH}_i${CURRENT_IMAGE}"
    
    python benchmark_breaches.py \
        case=1_single_image_small \
        attack=gradvit \
        attack.optim.step_size=$CURRENT_STEP \
        attack.regularization.total_variation.scale=$CURRENT_TV \
        attack.regularization.patch_prior.scale=$CURRENT_PATCH \
        attack.regularization.image_prior.scale=$CURRENT_IMAGE \
        dryrun=True > "logs/run_${i}_${param_tag}.log" 2>&1

done
