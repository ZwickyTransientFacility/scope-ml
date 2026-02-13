#!/bin/bash
#SBATCH --job-name=regen_ztf_chunk
#SBATCH --output=/home/mcoughli/scope-ml/regenerate_training/logs/regen_chunk_%A_%a.out
#SBATCH --error=/home/mcoughli/scope-ml/regenerate_training/logs/regen_chunk_%A_%a.err
#SBATCH --partition=milan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-7

# Phase 2 (parallel): Split 170K sources across 8 nodes.
# Each array task processes ~21K sources independently.
# Submit via: sbatch regenerate_training/run_phase2_array.sh
# After all tasks finish, merge with: python tools/merge_training_chunks.py

N_CHUNKS=8

module load gcc/13.2.0 python/3.11.5
export LD_LIBRARY_PATH=$(python3 -c "import sys, os; print(os.path.dirname(sys.executable) + '/../lib')"):$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

cd /home/mcoughli/scope-ml

echo "Array task ${SLURM_ARRAY_TASK_ID} of ${N_CHUNKS} starting on $(hostname) at $(date)"

/home/mcoughli/scope-ml/.venv/bin/python3 tools/regenerate_training_set.py \
    --old-training-set /home/mcoughli/training_set.parquet \
    --output /home/mcoughli/scope-ml/regenerate_training/chunk_${SLURM_ARRAY_TASK_ID}.parquet \
    --kowalski-cache /home/mcoughli/scope-ml/regenerate_training/kowalski_cache \
    --doCPU \
    --top-n-periods 50 \
    --max-freq 48.0 \
    --period-batch-size 1000 \
    --Ncore 16 \
    --doRemoveTerrestrial \
    --xmatch-radius-arcsec 2.0 \
    --chunk-index ${SLURM_ARRAY_TASK_ID} \
    --n-chunks ${N_CHUNKS}

echo "Chunk ${SLURM_ARRAY_TASK_ID} finished at $(date)"
