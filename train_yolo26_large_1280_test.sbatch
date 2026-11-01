#!/bin/bash

#SBATCH --job-name=yolo26_large_1280_v1
#SBATCH --partition=uri-gpu
#SBATCH --gres=gpu:a100:1

#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00

#SBATCH --output=/home/zachary_johnson_uri_edu/workflows/logs/yolo26_large_1280_v1_%j.out
#SBATCH --error=/home/zachary_johnson_uri_edu/workflows/logs/yolo26_large_1280_v1_%j.err

set -euo pipefail


echo "================================================="
echo "OceanDetect Training"
echo "YOLO26 Large @ 1280 - V1"
echo "================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo


PYTHON=/modules/opt/linux-ubuntu24.04-x86_64/jupyterlab/unity-jupyterlab4.5.2/bin/python

TRAIN_SCRIPT=/home/zachary_johnson_uri_edu/workflows/scripts/train_yolo26_large_1280_v1.py


echo "GPU:"
nvidia-smi

echo

echo "Python:"
$PYTHON --version

echo

echo "Ultralytics:"
$PYTHON -c "import ultralytics; print(ultralytics.__version__)"

echo

echo "CUDA:"
$PYTHON -c "
import torch

print('CUDA available:', torch.cuda.is_available())

if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('CUDA version:', torch.version.cuda)
"

echo
echo "================================================="
echo "Starting YOLO26 Large training"
echo "================================================="
echo


$PYTHON "$TRAIN_SCRIPT"


echo
echo "================================================="
echo "Training finished successfully"
echo "Finished: $(date)"
echo "================================================="
