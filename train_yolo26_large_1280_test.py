from pathlib import Path

from ultralytics import YOLO

# =========================================================
# OceanDetect Training Experiment
#
# Model: YOLO26 Large
# Image Size: 1280
# Dataset: fish_detection_5class_merged_v4
# Experiment: V1
# =========================================================


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

DATA_YAML = Path(
    "/work/pi_indrani_mandal_uri_edu/"
    "OceanDetect-Shared/"
    "datasets/"
    "fish_detection_5class_merged_v4/"
    "data.yaml"
)


# ---------------------------------------------------------
# My Training Output
# ---------------------------------------------------------

PROJECT_DIR = Path("/home/zachary_johnson_uri_edu/workflows/runs")

RUN_NAME = "ZJ_yolo26_large_1280_v1"


# ---------------------------------------------------------
# Validate Paths
# ---------------------------------------------------------

print("=================================================")
print("OceanDetect YOLO26 Large Training")
print("=================================================")

print(f"Dataset: {DATA_YAML}")
print(f"Output:  {PROJECT_DIR / RUN_NAME}")
print()


if not DATA_YAML.exists():
    raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")


PROJECT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

print("Loading YOLO26 Large...")

model = YOLO("yolo26l.pt")


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

results = model.train(
    data=str(DATA_YAML),
    epochs=200,
    patience=20,
    imgsz=1280,
    batch=16,
    optimizer="AdamW",
    lr0=0.001,
    warmup_epochs=3,
    weight_decay=0.0005,
    mosaic=0.5,
    close_mosaic=10,
    amp=True,
    cache=False,
    workers=8,
    seed=42,
    deterministic=True,
    device=0,
    project=str(PROJECT_DIR),
    name=RUN_NAME,
)


print()
print("=================================================")
print("Training complete")
print("=================================================")

print(f"Results saved to: {PROJECT_DIR / RUN_NAME}")
