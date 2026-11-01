# OceanDetect training

YOLO26 Large object-detection training scripts for the **Unity HPC platform**, used through URI Research Computing and SLURM. Includes the original baseline, a fixed augmentation experiment, and a reusable launcher that accepts dataset and training options at submission time.

For HPC configuration questions, use [Unity's official documentation](https://docs.unity.rc.umass.edu/documentation/), especially the [partition list](https://docs.unity.rc.umass.edu/documentation/cluster_specs/partitions/) and [GPU guide](https://docs.unity.rc.umass.edu/documentation/tools/gpus/). The `uri-gpu`/A100 requests in these scripts are saved experiment defaults; check Unity for current resources, access, and scheduler requirements. See our [Unity setup and reference links](docs/hpc-setup.md#official-unity-documentation) for the full starting point.

## Choose a launcher

- **For new runs:** [Reusable command-line launcher](docs/scripts/configurable-training.md) — choose the dataset, epochs, resolution, early stopping, batch size, and augmentation without editing Python.
- **For the fixed augmentation experiment:** [Augmentation V1](docs/scripts/augmentation-v1.md) — labeled constants with mixed color/grayscale, mild blur, and contrast enhancement.
- **For the original experiment:** [Baseline V1](docs/scripts/baseline-v1.md) — documents the original Python and sbatch files and their deployed paths.

Each guide explains both its Python script and its sbatch file. Training behavior is unchanged by this documentation update.

## Quick start

Read [Unity HPC setup](docs/hpc-setup.md), connect using Unity's [Quick Start](https://docs.unity.rc.umass.edu/documentation/get-started/quickstart/), activate your CUDA Python environment, and submit from the directory containing `train_yolo26.py` and `train_yolo26.sbatch`. Confirm your partition/GPU request against Unity's documentation and replace the dataset path:

```bash
sbatch --time=00:30:00 train_yolo26.sbatch \
  --data /absolute/path/to/data.yaml \
  --epochs 2 --imgsz 640 --batch 4 \
  --fraction 0.01 --close-mosaic 0 --name smoke_test
```

This tests two epochs using 1% of the training split. Validation still uses the full validation split, so duration varies. Increase the fraction for a small dataset. To inspect arguments without training dependencies or a GPU:

```bash
python3 train_yolo26.py --data /absolute/path/to/data.yaml --dry-run
python3 train_yolo26.py --help
```

The reusable launcher's defaults are YOLO26 L, 1280 pixels, batch 16, up to 200 epochs, patience 20, and mixed augmentation. The short example overrides these for a smoke test. Follow its output in `yolo26_JOBID.out` and check `yolo26_JOBID.err` for errors.

## Documentation

Start at the [documentation index](docs/README.md) for shared setup, script guides, and documentation organization. Keep general onboarding here, shared HPC instructions in `docs/`, and file-specific configuration and examples in `docs/scripts/`.

Training datasets, pretrained weights, environments, and run outputs are configured on the HPC; they are not bundled here. Full GPU training still needs validation on your HPC environment.
