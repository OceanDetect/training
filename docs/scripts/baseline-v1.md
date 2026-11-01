# Original baseline V1

[Documentation index](../README.md) · [Shared HPC setup](../hpc-setup.md) · [Reusable alternative](configurable-training.md)

This original experiment targets **Unity HPC** through URI. For current HPC configuration, partitions, and GPUs, consult Unity's [partition list](https://docs.unity.rc.umass.edu/documentation/cluster_specs/partitions/), [GPU guide](https://docs.unity.rc.umass.edu/documentation/tools/gpus/), and [batch-job guide](https://docs.unity.rc.umass.edu/documentation/jobs/sbatch/). See [shared Unity references](../hpc-setup.md#official-unity-documentation) for environment setup and support.

## Files and responsibilities

- [train_yolo26_large_1280_test.py](../../train_yolo26_large_1280_test.py) defines and immediately runs the original YOLO26 Large experiment.
- [train_yolo26_large_1280_test.sbatch](../../train_yolo26_large_1280_test.sbatch) requests resources, prints environment diagnostics, and runs the deployed Python file.

This pair is preserved as the original reference experiment. Configure it by editing constants and paths; it does not accept command-line training options.

## Python configuration

- `DATA_YAML`: `/work/pi_indrani_mandal_uri_edu/OceanDetect-Shared/datasets/fish_detection_5class_merged_v4/data.yaml`.
- `PROJECT_DIR`: `/home/zachary_johnson_uri_edu/workflows/runs`.
- `RUN_NAME`: `ZJ_yolo26_large_1280_v1`.
- Model loading uses `YOLO("yolo26l.pt")`. Use local weights in a deployed copy if compute nodes cannot download them.
- The `model.train(...)` call holds other settings: 200 epochs, patience 20, 1280 pixels, batch 16, AdamW, learning rate `0.001`, warmup 3, weight decay `0.0005`, mosaic `0.5`, and mosaic disabled for the final ten epochs.
- AMP and deterministic operations are enabled; caching is disabled. Workers are `8`, seed is `42`, and device is the first visible GPU (`0`).

The script checks dataset path existence, creates the project directory, and starts training. It uses standard Ultralytics augmentation, including optional default Albumentations transforms if installed. This is not an augmentation-free baseline.

## sbatch configuration

Saved defaults are partition `uri-gpu`, one A100, 16 CPUs, 60 GB memory, and a 24-hour limit. Confirm current partition access, GPU choices, and any account/QOS requirements in Unity's documentation before submission; these values describe the original experiment, not all currently available Unity resources.

`PYTHON` is currently `/modules/opt/linux-ubuntu24.04-x86_64/jupyterlab/unity-jupyterlab4.5.2/bin/python`. `TRAIN_SCRIPT` points to `/home/zachary_johnson_uri_edu/workflows/scripts/train_yolo26_large_1280_v1.py`.

That interpreter path is a saved deployment example. Use Unity's [PyTorch](https://docs.unity.rc.umass.edu/documentation/tools/pytorch/) and [module](https://docs.unity.rc.umass.edu/documentation/software/modules/) guidance when configuring the interpreter for a new deployment; no particular JupyterLab version is required by these scripts.

**The deployed filename intentionally differs from the repository filename.** Deploy the Python file at that location/name or adjust `TRAIN_SCRIPT` in your deployed sbatch. Dataset/output paths must be available on the compute node.

Log directives use `/home/zachary_johnson_uri_edu/workflows/logs`, with names `yolo26_large_1280_v1_JOBID.out` and `.err`. Edit the literal paths and create the parent directory before submission.

The sbatch uses strict error handling, prints `nvidia-smi`, Python/Ultralytics versions and CUDA status, then runs training. It does not change directories; relative model paths use the job working directory.

## Submit

After configuring and deploying the files:

```bash
mkdir -p /home/zachary_johnson_uri_edu/workflows/logs
sbatch train_yolo26_large_1280_test.sbatch
```

Use the configured log directory if changed. Adding `--epochs 2` after this sbatch filename will not change training settings: the original launcher does not forward arguments.

## Short test and output behavior

For a quick first run, use the [reusable smoke example](configurable-training.md#short-test-run). To test this exact baseline, use a separate deployed copy and sbatch pointing to it: set a distinct run name, reduce epochs to 2, set `close_mosaic=0`, and add `fraction=0.01` to `model.train(...)`. Increase the fraction if needed. Restore full settings before comparing model quality.

Ultralytics produces metrics, plots, and checkpoints under the selected project/run directory. It may choose a numbered directory on reruns. **The original final print always shows the requested path**, so use the Ultralytics log to find the actual output directory.

The original Python file has no main guard: importing it starts training. The sbatch prints CUDA availability but does not explicitly fail at that diagnostic if CUDA is unavailable; later GPU training setup reports errors. Its Python invocation is not explicitly unbuffered.

## Manual resume

There is no resume switch. For an interrupted baseline run, use the same environment and a separate sbatch copy; replace its training invocation with this block, setting the actual checkpoint:

```bash
"$PYTHON" -u - <<'PY'
from ultralytics import YOLO

model = YOLO("/absolute/path/to/baseline_run/weights/last.pt")
model.train(resume=True)
print(f"Results saved to: {model.trainer.save_dir}")
PY
```

Do not import the original script to resume. Keep checkpoint dataset/output paths accessible and preserve the environment's augmentation behavior. No automatic requeue is configured.
