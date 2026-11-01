# Reusable command-line training

[Documentation index](../README.md) · [Shared HPC setup](../hpc-setup.md)

This launcher is used on **Unity HPC**. For cluster configuration questions, check Unity's [partition list](https://docs.unity.rc.umass.edu/documentation/cluster_specs/partitions/), [GPU guide](https://docs.unity.rc.umass.edu/documentation/tools/gpus/), and [batch-job documentation](https://docs.unity.rc.umass.edu/documentation/jobs/sbatch/). See [shared Unity setup](../hpc-setup.md#official-unity-documentation) for software and support links.

## Files and responsibilities

- [train_yolo26.py](../../train_yolo26.py) parses and validates training options, prints effective settings, checks the dataset and CUDA, builds mixed augmentation when selected, and launches training.
- [train_yolo26.sbatch](../../train_yolo26.sbatch) requests resources, selects the interpreter and deployed Python file, and forwards trailing arguments unchanged. It preserves the Python process's exit status.

Use `train_yolo26.py` and `train_yolo26.sbatch` to choose a dataset and common settings at submission time. Both earlier script pairs remain available unchanged.

## One-time setup

1. Keep the new Python and sbatch files together in an accessible directory, and submit from that directory. If the Python file is elsewhere, set `TRAIN_SCRIPT=/absolute/path/train_yolo26.py` when submitting.
2. Configure and activate your CUDA Python environment using Unity's [PyTorch](https://docs.unity.rc.umass.edu/documentation/tools/pytorch/) and [module](https://docs.unity.rc.umass.edu/documentation/software/modules/) guidance, or put the required activation commands in the sbatch's labeled environment section. It uses `python3` from that environment by default; set `PYTHON=/absolute/path/to/python` to select another interpreter.
3. Confirm the scheduler section against Unity's current partition/GPU guidance and your access. Saved defaults request one A100 on `uri-gpu`, 16 CPUs, 60 GB RAM, and 24 hours. Logs default to `yolo26_JOBID.out` and `.err` in the submission directory. Training outputs default to `runs/` there. No user or dataset paths are hardcoded in this pair.

## Short test run

First check the arguments locally without loading training dependencies or writing files:

```bash
python3 train_yolo26.py --data /absolute/path/to/data.yaml \
  --epochs 2 --imgsz 640 --batch 4 --fraction 0.01 \
  --close-mosaic 0 --name smoke_test --dry-run
```

Then submit the same settings to a GPU node (replace the dataset path):

```bash
sbatch --time=00:30:00 train_yolo26.sbatch \
  --data /absolute/path/to/data.yaml \
  --epochs 2 --imgsz 640 --batch 4 --fraction 0.01 \
  --close-mosaic 0 --name smoke_test
```

This trains for two epochs on 1% of the training images with mixed grayscale/blur/contrast augmentation. Validation still uses the full validation split, so duration depends on dataset size. Increase `--fraction` if 1% leaves too few images. The dry run checks arguments and prints settings; the GPU run checks actual dataset paths, dependencies, CUDA, and augmentation initialization.

Use the job ID printed by `sbatch` to follow its log:

```bash
tail -f yolo26_JOBID.out
```

Check the `.err` log for errors as well. Successful completion prints the actual results directory. Inspect training-batch plots, `results.csv`, and `weights/best.pt` / `weights/last.pt`. This smoke run verifies that training works; it does not measure model quality. For a production memory check, repeat with `--imgsz 1280 --batch 16` while retaining the short epoch count and subset.

## Full run with explicit settings

```bash
sbatch --job-name=fish_aug_v2 train_yolo26.sbatch \
  --data /absolute/path/to/data.yaml \
  --project /absolute/path/to/runs --name fish_aug_v2 \
  --model /absolute/path/to/yolo26l.pt \
  --epochs 200 --imgsz 1280 --patience 20 \
  --batch 16 --workers 8 --seed 42 \
  --optimizer AdamW --lr0 0.001 \
  --augmentation mixed --grayscale 0.20 --blur 0.10 --clahe 0.10
```

Use spaces between arguments, with no commas. **SLURM options go before the sbatch filename; training options go after it.** For example, `--time` and `--cpus-per-task` configure SLURM, while `--epochs` and `--workers` configure training. Keep workers within the allocated CPU count. `--patience 20` stops after 20 epochs without validation improvement; `--patience 0` disables early stopping. `--optimizer auto` lets Ultralytics choose the optimizer and learning rate rather than honoring `--lr0` as a fixed setting.

Run `python3 train_yolo26.py --help` for all options and defaults, including mosaic, warmup, weight decay, AMP, and deterministic operation. `--augmentation standard` retains standard Ultralytics augmentation, including its optional default Albumentations transforms when installed. It does not disable augmentation. `mixed` is the default and uses the same custom transform strengths as augmentation V1. New output directories are numbered on name collisions. The launcher has no `--resume` flag. See manual resume below for interrupted runs.

To use a different interpreter or deployed script without editing the launcher:

```bash
PYTHON=/path/to/python TRAIN_SCRIPT=/path/to/train_yolo26.py \
  sbatch train_yolo26.sbatch --data /path/to/data.yaml --epochs 200
```

Relative dataset, model, and output paths are interpreted from the submission working directory. Dataset YAML entries must themselves resolve to accessible images/labels. Absolute paths are easiest to reuse across jobs. Scheduler directives may be overridden before the filename; choose actual partition and GPU values from [Unity's GPU guide](https://docs.unity.rc.umass.edu/documentation/tools/gpus/) rather than substituting an assumed partition name. See [shared dependencies](../hpc-setup.md#dependencies); jobs never install packages.

## Python argument reference

`--data` is required. All other options have defaults. `--help` needs no training dependencies.

### Dataset and output

- `--data PATH`: required dataset YAML; expands `~` and resolves to an absolute path.
- `--model PATH_OR_NAME`: default `yolo26l.pt`. Use absolute weights paths for offline compute nodes.
- `--project PATH`: default `runs/` in the job working directory; expanded and resolved to an absolute path.
- `--name TEXT`: default `<model-stem>_<augmentation>_<imgsz>`, such as `yolo26l_mixed_1280`.

### Training

- `--epochs`: maximum epochs, default `200`; integer at least 1.
- `--imgsz`: square training resolution, default `1280`; positive integer.
- `--patience`: early-stopping patience, default `20`; nonnegative integer, with `0` disabling early stopping.
- `--batch`: images per batch, default `16`; positive integer. Automatic batch sizing is not exposed.
- `--workers`: data loader workers, default `8`; nonnegative integer.
- `--seed`: random seed, default `42`; nonnegative integer.
- `--optimizer`: `AdamW` (default), `SGD`, or `auto`.
- `--lr0`: initial learning rate, default `0.001`; finite positive number. `auto` selects its own rate.
- `--weight-decay`: default `0.0005`; value from 0 to 1.
- `--warmup-epochs`: default `3`; nonnegative integer.
- `--fraction`: training subset fraction, default `1.0`; greater than 0 and no more than 1. Validation is not subsetted.
- `--no-amp`: disable mixed precision; enabled by default.
- `--no-deterministic`: disable deterministic operations; enabled by default.

Caching stays disabled, training uses the first visible GPU (`device=0`), and this launcher supports one GPU. Increasing the sbatch GPU request alone does not enable distributed training.

### Augmentation and inspection

- `--augmentation`: `mixed` (default) replaces the optional Albumentations list with custom transforms; `standard` uses Ultralytics defaults.
- `--grayscale`: probability in mixed mode, default `0.20`; three identical output channels.
- `--blur`: probability in mixed mode, default `0.10`; kernel range `(3, 5)`.
- `--clahe`: probability in mixed mode, default `0.10`; clip limit `(1.0, 2.0)`.
- `--mosaic`: probability, default `0.5`, in either mode.
- `--close-mosaic`: disable mosaic in the final N epochs, default `10`; nonnegative integer. Set `0` for the two-epoch smoke example to retain mosaic.
- `--dry-run`: print settings and exit without training dependencies, GPU checks, dataset validation, or file writes.

Probability arguments accept values from 0 to 1. Grayscale, blur, and CLAHE probabilities apply independently in mixed mode, modifying pixels while preserving annotations. Standard YOLO HSV, geometry, and flip augmentation remain active. Setting the three custom probabilities to zero does not disable standard transformations.

## sbatch configuration details

The labeled scheduler block holds partition/GPU/CPU/memory/time settings. Configure it using Unity's documentation or override scheduler options before the filename. `PYTHON` defaults to `python3` on `PATH`. `TRAIN_SCRIPT` defaults to `train_yolo26.py` in `SLURM_SUBMIT_DIR`. Set either variable to an absolute deployed path when needed.

The launcher does not change directories. If using SLURM `--chdir`, use explicit absolute paths and `TRAIN_SCRIPT` to keep lookup and data/output resolution unambiguous.

## Manual resume

There is no command-line resume option. Use the same environment and a separate sbatch copy with the same resource/environment setup. Replace its final `exec` command with this block, using the actual interrupted checkpoint and the same mixed-mode probabilities as that job:

```bash
"$PYTHON" -u - <<'PY'
import albumentations as A
from ultralytics import YOLO

model = YOLO("/absolute/path/to/run/weights/last.pt")
model.train(
    resume=True,
    augmentations=[
        A.ToGray(num_output_channels=3, p=0.20),
        A.Blur(blur_limit=(3, 5), p=0.10),
        A.CLAHE(clip_limit=(1.0, 2.0), p=0.10),
    ],
)
print(f"Results saved to: {model.trainer.save_dir}")
PY
```

For an interrupted `standard` run, omit the Albumentations import and the `augmentations` argument. Resume uses checkpoint training state/settings, including paths and epoch progress. Keep the original dataset/output paths accessible and check the logged transform list. Jobs are not automatically requeued.
