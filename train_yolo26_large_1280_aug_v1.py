"""YOLO26 L augmentation experiment. Adjust the configuration below before submission."""

from pathlib import Path
import sys

import albumentations as A
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import Albumentations


# =========================================================
# CONFIGURATION: Environment and paths — adjust for your HPC
# =========================================================

DATA_YAML = Path(
    "/work/pi_indrani_mandal_uri_edu/OceanDetect-Shared/"
    "datasets/fish_detection_5class_merged_v4/data.yaml"
)
PROJECT_DIR = Path("/home/zachary_johnson_uri_edu/workflows/runs")
RUN_NAME = "ZJ_yolo26_large_1280_aug_v1"
# Use an absolute path to pre-downloaded weights if compute nodes lack internet.
# A model filename alone allows Ultralytics to download missing weights.
MODEL_WEIGHTS = "yolo26l.pt"


# =========================================================
# CONFIGURATION: Training — baseline settings for comparison
# =========================================================

TRAINING = {
    "epochs": 200,
    "patience": 20,
    "imgsz": 1280,
    "batch": 16,  # Reduce if GPU memory is insufficient; record comparison changes.
    "optimizer": "AdamW",
    "lr0": 0.001,
    "warmup_epochs": 3,
    "weight_decay": 0.0005,
    "mosaic": 0.5,
    "close_mosaic": 10,
    "amp": True,
    "cache": False,
    "workers": 8,  # Keep within the CPUs allocated by sbatch.
    "seed": 42,
    "deterministic": True,
    "device": 0,  # First GPU visible inside the SLURM allocation.
}


# =========================================================
# CONFIGURATION: Augmentation — training images only
# =========================================================

GRAYSCALE_PROBABILITY = 0.20  # Keep three channels for the pretrained color model.
BLUR_PROBABILITY = 0.10
BLUR_LIMIT = (3, 5)  # Mild blur: odd kernel sizes in pixels.
CLAHE_PROBABILITY = 0.10  # Occasional local contrast enhancement.
CLAHE_CLIP_LIMIT = (1.0, 2.0)

# This list replaces the optional default Albumentations transforms.
# Standard YOLO HSV, geometry, flip, and mosaic augmentation remain active.
# Transforms have independent probabilities and may occur together.
def build_augmentations():
    return [
        A.ToGray(num_output_channels=3, p=GRAYSCALE_PROBABILITY),
        A.Blur(blur_limit=BLUR_LIMIT, p=BLUR_PROBABILITY),
        A.CLAHE(clip_limit=CLAHE_CLIP_LIMIT, p=CLAHE_PROBABILITY),
    ]


def main():
    print("=================================================")
    print("OceanDetect YOLO26 Large @ 1280 — Augmentation V1")
    print("=================================================")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Ultralytics: {ultralytics.__version__}")
    print(f"Albumentations: {A.__version__}")
    print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")
    print(f"Dataset: {DATA_YAML}")
    print(f"Weights: {MODEL_WEIGHTS}")
    print(f"Requested output: {PROJECT_DIR / RUN_NAME}")
    print(f"Training settings: {TRAINING}")

    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the GPU allocation and Python environment.")
    print(f"GPU: {torch.cuda.get_device_name(TRAINING['device'])}")

    augmentations = build_augmentations()
    # Check API support and initialization before loading weights. Ultralytics
    # can otherwise log an Albumentations setup error and skip augmentation.
    get_cfg(overrides={"augmentations": augmentations})
    if Albumentations(p=1.0, transforms=augmentations).transform is None:
        raise RuntimeError("Custom augmentation initialization failed. Check dependency versions.")
    print("Custom augmentation settings:")
    for transform in augmentations:
        print(f"  {transform}")

    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(MODEL_WEIGHTS)
    model.train(
        data=str(DATA_YAML),
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        exist_ok=False,  # Number reruns instead of overwriting existing results.
        augmentations=augmentations,
        **TRAINING,
    )

    print("Training complete")
    print(f"Results saved to: {model.trainer.save_dir}")


if __name__ == "__main__":
    main()
