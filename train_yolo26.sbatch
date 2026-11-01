#!/bin/bash

# CONFIGURATION: HPC resources — adjust once for your cluster.
# Override at submission, e.g. sbatch --time=00:30:00 train_yolo26.sbatch ...
#SBATCH --job-name=yolo26
#SBATCH --partition=uri-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00

# Logs go to the submission directory; no hardcoded user paths.
# Override with sbatch --output=/existing/path/%x_%j.out --error=... ...
#SBATCH --output=yolo26_%j.out
#SBATCH --error=yolo26_%j.err

set -euo pipefail

# CONFIGURATION: Python environment — activate your CUDA environment before
# submission, or add site-required module loads / activation commands here.
# Optionally submit with: PYTHON=/path/to/python sbatch train_yolo26.sbatch ...
PYTHON="${PYTHON:-python3}"

# Submit from the directory containing train_yolo26.py, or set TRAIN_SCRIPT to
# its absolute deployed path. SLURM copies the sbatch file, not the Python file.
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${SLURM_SUBMIT_DIR:-$PWD}/train_yolo26.py}"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Python: $PYTHON"
echo "Training script: $TRAIN_SCRIPT"

if ! command -v "$PYTHON" > /dev/null 2>&1; then
    echo "Python executable not found: $PYTHON" >&2
    exit 1
fi
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "Training script not found: $TRAIN_SCRIPT (set TRAIN_SCRIPT or submit from its directory)" >&2
    exit 1
fi

# Preserve the submission working directory for relative dataset/output paths.
# Every argument after the sbatch filename is forwarded intact to Python.
exec "$PYTHON" -u "$TRAIN_SCRIPT" "$@"
