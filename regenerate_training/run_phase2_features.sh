#!/bin/bash
#SBATCH --job-name=regen_ztf_ts
#SBATCH --output=/home/mcoughli/scope-ml/regenerate_training/logs/regen_%j.out
#SBATCH --error=/home/mcoughli/scope-ml/regenerate_training/logs/regen_%j.err
#SBATCH --partition=milan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# Phase 2: Run GPU feature generation using pre-downloaded Kowalski cache.
# Submit via: sbatch regenerate_training/run_phase2_features.sh

module load gcc/13.2.0 python/3.11.5
export LD_LIBRARY_PATH=$(python3 -c "import sys, os; print(os.path.dirname(sys.executable) + '/../lib')"):$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

cd /home/mcoughli/scope-ml

echo "Starting on $(hostname) at $(date)"

/home/mcoughli/scope-ml/.venv/bin/python3 tools/regenerate_training_set.py \
    --old-training-set /home/mcoughli/training_set.parquet \
    --output /home/mcoughli/scope-ml/regenerate_training/new_training_set.parquet \
    --kowalski-cache /home/mcoughli/scope-ml/regenerate_training/kowalski_cache \
    --doCPU \
    --top-n-periods 50 \
    --max-freq 48.0 \
    --period-batch-size 1000 \
    --Ncore 16 \
    --doRemoveTerrestrial \
    --xmatch-radius-arcsec 2.0

echo "Finished at $(date)"
