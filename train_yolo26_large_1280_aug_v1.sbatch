#!/bin/bash

# =========================================================
# CONFIGURATION: Scheduler resources — adjust for your HPC
# =========================================================
#SBATCH --job-name=yolo26_large_1280_aug_v1
#SBATCH --partition=uri-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00

# =========================================================
# CONFIGURATION: Log paths — use literal, existing directories
# Create these directories BEFORE sbatch; shell variables do not expand here.
# =========================================================
#SBATCH --output=/home/zachary_johnson_uri_edu/workflows/logs/yolo26_large_1280_aug_v1_%j.out
#SBATCH --error=/home/zachary_johnson_uri_edu/workflows/logs/yolo26_large_1280_aug_v1_%j.err

set -euo pipefail

# =========================================================
# CONFIGURATION: Python environment — adjust for your HPC
# Add any required module loads / environment activation here.
# This interpreter must have CUDA PyTorch, Ultralytics, and Albumentations.
# =========================================================
PYTHON=/modules/opt/linux-ubuntu24.04-x86_64/jupyterlab/unity-jupyterlab4.5.2/bin/python

# =========================================================
# CONFIGURATION: Deployed script path — adjust for your HPC
# This may differ from the location/name in the shared repository.
# =========================================================
TRAIN_SCRIPT=/home/zachary_johnson_uri_edu/workflows/scripts/train_yolo26_large_1280_aug_v1.py

echo "================================================="
echo "OceanDetect YOLO26 Large @ 1280 — Augmentation V1"
echo "================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Python: $PYTHON"
echo "Training script: $TRAIN_SCRIPT"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python executable not found: $PYTHON" >&2
    exit 1
fi
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "Training script not found: $TRAIN_SCRIPT" >&2
    exit 1
fi

nvidia-smi
# Give relative model-weight filenames a predictable working directory.
cd "$(dirname "$TRAIN_SCRIPT")"
"$PYTHON" -u "$TRAIN_SCRIPT"

echo "Training finished successfully"
echo "Finished: $(date)"
