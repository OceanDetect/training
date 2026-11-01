# Unity HPC setup and operations

[Documentation index](README.md) · [Repository README](../README.md)

Use this guide for OceanDetect training on **Unity HPC**, common to all script pairs. Follow the relevant [script guide](README.md#script-guides) for repository-specific paths, configuration, and commands.

## Official Unity documentation

Unity's documentation is the source of truth for HPC configuration, available partitions and GPUs, access requirements, module/environment setup, and scheduling policies. This repository documents training scripts and saved experiment settings; it does not maintain a separate cluster inventory or policy reference.

- [Unity documentation home](https://docs.unity.rc.umass.edu/documentation/): start here for platform questions.
- [Unity Quick Start](https://docs.unity.rc.umass.edu/documentation/get-started/quickstart/): get started and connect to the platform.
- [Partition list](https://docs.unity.rc.umass.edu/documentation/cluster_specs/partitions/): choose partitions and check their access requirements and limits.
- [GPU guide](https://docs.unity.rc.umass.edu/documentation/tools/gpus/): choose and request GPU resources, including constraints and CUDA considerations.
- [GPU summary](https://docs.unity.rc.umass.edu/documentation/cluster_specs/gpu_summary/) and [node list](https://docs.unity.rc.umass.edu/nodes/): check documented hardware and partition membership.
- [Batch jobs](https://docs.unity.rc.umass.edu/documentation/jobs/sbatch/): configure and submit SLURM jobs on Unity.
- [Environment modules](https://docs.unity.rc.umass.edu/documentation/software/modules/) and [PyTorch](https://docs.unity.rc.umass.edu/documentation/tools/pytorch/): configure the software environment for training.
- [How to ask Unity for help](https://docs.unity.rc.umass.edu/documentation/help/asking-questions/): follow this for unresolved platform, access, or scheduler questions.

For URI-specific onboarding, also see [URI's Using Unity page](https://its.uri.edu/research-computing/using-unity/). Use the current official Unity documentation when choosing resources or changing a job configuration.

Before submitting, connect to Unity, make this repository and the dataset available there, configure your Python environment using Unity's guidance, and confirm that the chosen partition/GPU request is appropriate for your account. The examples assume submission from Unity with access to the configured shared files.

## Dependencies

Follow Unity's [PyTorch setup](https://docs.unity.rc.umass.edu/documentation/tools/pytorch/) and [module guidance](https://docs.unity.rc.umass.edu/documentation/software/modules/) for the CUDA/Python environment. For these scripts, use an Ultralytics version supporting YOLO26 and the Python `augmentations` argument, and Albumentations supporting `ToGray(num_output_channels=3)` (the documented custom-transform examples require Albumentations 1.4.22 or newer). See the [Ultralytics Albumentations integration](https://docs.ultralytics.com/integrations/albumentations/).

To add Albumentations to your own writable environment, run this once using the configured interpreter (substitute its actual path):

```bash
/path/to/python -m pip install 'albumentations>=1.4.22,<3'
```

Keep the CUDA/PyTorch setup compatible with Unity's guidance. Interpreter paths under `/modules/` in the fixed scripts are saved deployment examples, not a requirement to use that particular JupyterLab installation or version. Use the same package versions for baseline and augmentation comparisons; record the working environment with `/path/to/python -m pip freeze > training-environment.txt`. This repository does not yet have an HPC-validated version lock.

The reusable launcher's `--help` and `--dry-run` need no training packages. Actual training requires the CUDA environment. Albumentations is required for mixed augmentation and fixed augmentation V1; standard/baseline training can run without it. Installing it can change the baseline's optional augmentation, so keep environments consistent when comparing runs.

The augmentation runners log versions and check the dataset file, CUDA, API support, and custom-transform initialization. Standard mode checks the dataset file and CUDA without constructing custom transforms. The original baseline has fewer explicit checks; see its guide.

## Dataset, weights, and storage

- Use an accessible detection dataset YAML with the intended classes and train/validation splits. Confirm its image/label paths resolve on compute nodes. The Python preflight checks YAML file existence; Ultralytics validates contents and dataset paths afterward.
- Prefer absolute paths for data, weights, and output storage. Ensure output storage is writable and has space for checkpoints and plots.
- Pre-download weights and use an absolute path if compute nodes lack internet. A filename alone can trigger a download if missing.
- `--fraction` limits training images only. Full validation may dominate a smoke run; increase a tiny fraction if it selects too few images.
- Datasets and weights are not included here. Real data loading, package compatibility, and GPU memory fit require an HPC smoke run.

## Scheduler and environment

Use Unity's [partition list](https://docs.unity.rc.umass.edu/documentation/cluster_specs/partitions/), [GPU guide](https://docs.unity.rc.umass.edu/documentation/tools/gpus/), and [batch-job guide](https://docs.unity.rc.umass.edu/documentation/jobs/sbatch/) to configure partition, GPU type, CPUs, memory, time limit, and any required account/QOS. The scripts currently request one A100 on `uri-gpu`, 16 CPUs, 60 GB RAM, and 24 hours. These are saved defaults, not a statement of current availability or access for every Unity user. GPU 0 means the first GPU visible within the allocation.

Use the environment configured following Unity's software guidance. The reusable launcher uses `python3` from the environment with a `PYTHON` override; fixed launchers contain interpreter paths to adjust. Keep workers within allocated CPUs. Batch size and resolution affect GPU memory use.

Scheduler options go before the filename; reusable training options go after it:

```bash
sbatch --time=00:30:00 --cpus-per-task=8 train_yolo26.sbatch \
  --data /absolute/path/to/data.yaml --workers 4 --epochs 2
```

Separate command-line arguments with spaces; Unity may document comma-separated values within a particular scheduler option such as a partition list. Shell variables are not expanded inside `#SBATCH` directives; use literal paths or submission-time overrides. `%j` expands to the job ID. Custom log directories must exist before submission because SLURM opens logs before the script runs. Start with [Unity's batch-job guide](https://docs.unity.rc.umass.edu/documentation/jobs/sbatch/); the upstream [sbatch reference](https://slurm.schedmd.com/sbatch.html) provides additional command syntax.

The reusable launcher writes logs in the submission directory by default. Fixed launchers use configured absolute log paths. See their guides for working-directory differences.

## Monitor jobs and inspect outputs

Replace `JOBID` with the value printed by `sbatch`:

```bash
squeue -j JOBID
tail -f yolo26_JOBID.out
```

The `tail` example uses the reusable launcher's default log. Use configured paths for fixed scripts, and check `.err` as well. If accounting is available, inspect a completed job with:

```bash
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode
```

Check completion and the actual Ultralytics save directory. Outputs normally include `results.csv`, training/validation plots, `weights/last.pt`, and `weights/best.pt`. Reruns may get numbered directories; the baseline's final print does not reflect the suffix.

Inspect training-batch plots for sensible boxes and realistic augmentation. A tiny random run may not show every transform. The [augmentation guide](scripts/augmentation-v1.md#smoke-check-and-comparison) includes a direct three-channel grayscale check.

## Common problems

For platform configuration, partition/GPU availability, scheduling, or access questions, consult the [official Unity references](#official-unity-documentation) first. Use Unity's [support guidance](https://docs.unity.rc.umass.edu/documentation/help/asking-questions/) if the documentation does not resolve the issue.

- **Python/script not found:** check `PYTHON` and `TRAIN_SCRIPT`. For the reusable launcher, submit from the Python file's directory or set its absolute path.
- **Missing package/unsupported augmentation argument:** check the interpreter and versions. Set up packages ahead of submission in your own environment; jobs do not install them.
- **CUDA unavailable:** check the GPU request and selected PyTorch environment; actual training needs a GPU allocation.
- **Images missing despite an existing YAML:** inspect YAML paths and shared storage availability on compute nodes.
- **GPU out of memory:** reduce batch size first. Test production resolution/batch separately before full training.
- **Missing logs:** check job state, configured paths, and whether parent directories existed at submission.
- **Smoke test taking longer than expected:** full validation still runs; queue time and possible weight downloads add time.
- **Time limit reached:** use the relevant guide's manual resume instructions and actual `last.pt`. No automatic requeue is configured.

## Compare experiments consistently

Use the same dataset version/split, pretrained weights, seed, software environment, and core settings. Record any changes in resolution, batch size, fraction, or optimizer. Smoke settings verify execution; restore full settings before evaluating model quality.

Compare aggregate and per-class precision, recall, and mAP. Preserve logs, arguments, checkpoints, and package versions to trace results to configurations. Augmentation strengths are starting values to evaluate rather than guaranteed improvements.
