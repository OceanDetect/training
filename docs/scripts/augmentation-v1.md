# Fixed augmentation V1

[Documentation index](../README.md) · [Shared HPC setup](../hpc-setup.md) · [Reusable alternative](configurable-training.md)

This experiment runs on **Unity HPC**. Use Unity's [partition list](https://docs.unity.rc.umass.edu/documentation/cluster_specs/partitions/), [GPU guide](https://docs.unity.rc.umass.edu/documentation/tools/gpus/), and [batch-job guide](https://docs.unity.rc.umass.edu/documentation/jobs/sbatch/) for current HPC configuration and resource questions. The [shared Unity guide](../hpc-setup.md#official-unity-documentation) links software setup and support.

## Files and responsibilities

- [train_yolo26_large_1280_aug_v1.py](../../train_yolo26_large_1280_aug_v1.py) contains the experiment constants and training entry point.
- [train_yolo26_large_1280_aug_v1.sbatch](../../train_yolo26_large_1280_aug_v1.sbatch) contains scheduler resources, interpreter, deployed script path, and logs.

Use this pair for a fixed experiment configured by editing constants. It does not parse command-line options. For `sbatch ... --data ... --epochs ...`, use the [reusable launcher](configurable-training.md).

## Python configuration and behavior

The environment/path section contains `DATA_YAML`, `PROJECT_DIR`, `RUN_NAME`, and `MODEL_WEIGHTS`. Defaults use the shared `fish_detection_5class_merged_v4` dataset, `/home/zachary_johnson_uri_edu/workflows/runs`, run name `ZJ_yolo26_large_1280_aug_v1`, and `yolo26l.pt`.

`TRAINING` matches the baseline's core settings: 200 epochs, patience 20, 1280 pixels, batch 16, AdamW at `0.001`, three warmup epochs, weight decay `0.0005`, mosaic `0.5` closed for the final ten epochs, eight workers, seed 42, AMP and deterministic operations enabled, caching disabled, and GPU 0.

The augmentation section exposes `GRAYSCALE_PROBABILITY=0.20`, `BLUR_PROBABILITY=0.10`, `BLUR_LIMIT=(3, 5)`, `CLAHE_PROBABILITY=0.10`, and `CLAHE_CLIP_LIMIT=(1.0, 2.0)`. Grayscale retains three channels. This list replaces the optional default Albumentations list; standard YOLO augmentation remains active. The transforms are independent, may overlap, and preserve annotations. Validation retains its original appearance; no augmented dataset copies are written.

The script checks the dataset file, CUDA, API support, and augmentation initialization before loading weights. It logs versions/settings and prints the actual output directory. Its main guard allows importing `build_augmentations()` without starting training; imports still require the training dependencies.

## Configure Python and sbatch for your environment

Edit the labeled sections before deploying/submitting. All settings are constants; the job needs no interactive input.

1. **Python: environment and paths.** Set `DATA_YAML` to an accessible detection dataset YAML, with valid train/validation paths and class names. Set `PROJECT_DIR` to writable storage, choose a distinct `RUN_NAME`, and configure `MODEL_WEIGHTS`. Use an absolute path to pre-downloaded weights if compute nodes lack internet. With a filename such as `yolo26l.pt`, missing weights may be downloaded into the working directory.
2. **Python: training.** Review `TRAINING`, especially batch size and workers. Defaults match the baseline: 1280 pixels, batch 16, 200 epochs, patience 20, AdamW, and eight workers. A100 memory capacity varies; confirm batch size with a smoke run.
3. **Python: augmentation.** Adjust the clearly labeled probabilities and strengths if needed. Grayscale retains three channels for the pretrained model. These are starting values to evaluate, not an accuracy guarantee.
4. **sbatch: scheduler resources.** Follow Unity's current partition/GPU/batch-job documentation to set resources and any required account/QOS. Saved defaults request one A100 on `uri-gpu`, 16 CPUs, 60 GB RAM, and 24 hours; confirm these match your Unity access and needs.
5. **sbatch: Python environment.** Set `PYTHON` and activation commands using Unity's [PyTorch](https://docs.unity.rc.umass.edu/documentation/tools/pytorch/) and [module](https://docs.unity.rc.umass.edu/documentation/software/modules/) guidance. The saved JupyterLab interpreter path is a deployment example, not a required Unity version. Configure dependencies before submission; the job does not install anything.
6. **sbatch: deployed script and logs.** Set `TRAIN_SCRIPT` to the deployed Python file. Set both log directives to literal absolute paths and create their parent directories **before** submitting. SLURM does not expand shell variables in `#SBATCH` directives. `%j` expands to the job ID.

The job changes to the deployed script's directory before launching Python. Relative model paths resolve there; prefer absolute dataset/output paths.

The sbatch uses strict error handling, checks the interpreter and script, prints GPU status with `nvidia-smi`, and launches Python with `-u` for unbuffered logs. Its default job name is `yolo26_large_1280_aug_v1`; log names include that experiment name and the job ID. See [shared dependencies](../hpc-setup.md#dependencies) before submission.

## Submit

Deploy the Python and sbatch files to the paths you configured, then run (example default log directory):

```bash
mkdir -p /home/zachary_johnson_uri_edu/workflows/logs
sbatch train_yolo26_large_1280_aug_v1.sbatch
```

Training runs unattended, with unbuffered logs and normal Ultralytics checkpointing (`weights/last.pt` and `weights/best.pt`). A failed preflight or training command causes the job to exit with an error. Repeated submissions create numbered result directories rather than overwriting an earlier run.

## Smoke check and comparison

- Check syntax with `bash -n train_yolo26_large_1280_aug_v1.sbatch` and compile the Python source.
- For a short HPC smoke run, use a separate deployed copy with a distinct run name, set `epochs=2`, `close_mosaic=0`, and add `fraction=0.01` to `TRAINING` (increase the fraction for a small dataset). Leave resolution and batch size at the intended production values to check memory fit. Use a shorter walltime if appropriate for your cluster.
- Confirm active transforms in logs, three-channel grayscale examples in the saved training-batch plots, sensible boxes, validation metrics, and saved checkpoints. A short random run may not show every transform; validate grayscale directly in the configured environment if needed:

  ```python
  import albumentations as A
  import numpy as np

  image = np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype=np.uint8)
  gray = A.ToGray(num_output_channels=3, p=1.0)(image=image)["image"]
  assert gray.shape == image.shape
  assert np.array_equal(gray[..., 0], gray[..., 1])
  assert np.array_equal(gray[..., 1], gray[..., 2])
  ```

- Restore the full settings before the production run. Compare aggregate and per-class validation precision, recall, and mAP against the baseline using the same split, seed, and software environment. These additional transforms change pixels only, so they preserve boxes/classes. Inspect the training-batch plots to confirm the complete YOLO pipeline still produces sensible labels.

## Resume after a walltime interruption

Resume an **interrupted augmentation run** from its actual `weights/last.pt`, using the same environment and dataset paths. In a separate copy of the sbatch, replace the final Python launch with the following, setting the checkpoint path explicitly. Keep the configured working directory so the deployed module can be imported:

```bash
"$PYTHON" -u - <<'PY'
from ultralytics import YOLO
from train_yolo26_large_1280_aug_v1 import build_augmentations

model = YOLO("/absolute/path/to/augmentation_run/weights/last.pt")
model.train(resume=True, augmentations=build_augmentations())
print(f"Results saved to: {model.trainer.save_dir}")
PY
```

If you renamed the deployed Python file, adjust the import accordingly. Resume restores training state and settings from the checkpoint, including optimizer and epoch; it does not start a new experiment. Confirm the logged transforms match the interrupted run. Jobs are not automatically requeued.
