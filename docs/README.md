# Documentation index

[Repository README](../README.md)

These guides describe OceanDetect training on **Unity HPC**. Use [Unity's official documentation](https://docs.unity.rc.umass.edu/documentation/) for cluster configuration, current partitions, GPU resources, access, and support. Our [Unity reference links](hpc-setup.md#official-unity-documentation) connect each topic to its official guide; the script guides describe this repository's saved settings and behavior.

## Shared instructions

- [Unity HPC setup and operations](hpc-setup.md): official Unity references, dependencies, data and weights, scheduler resources, monitoring, outputs, troubleshooting, and experiment comparisons.

## Script guides

- [Reusable command-line launcher](scripts/configurable-training.md): `train_yolo26.py` and `train_yolo26.sbatch`. Start here for new datasets and runs configured through `sbatch` arguments.
- [Fixed augmentation V1](scripts/augmentation-v1.md): `train_yolo26_large_1280_aug_v1.py` and its sbatch. Covers constants, transformations, smoke testing, and manual resume.
- [Original baseline V1](scripts/baseline-v1.md): `train_yolo26_large_1280_test.py` and its sbatch. Covers original defaults, deployed paths, and rerun behavior.

Each guide has separate Python and sbatch details so both files are documented together with a complete submission example.

## Where documentation belongs

- Keep the root README focused on the repository purpose, choosing a launcher, and a short first-run example.
- Keep shared environment, scheduler, and result-handling instructions in `docs/`. Link to shared instructions rather than repeating setup in each script guide.
- Keep file-specific settings, defaults, examples, and limitations in `docs/scripts/`, with relative links to the source files.
- When adding a launcher, add its guide here and an entry in the root README. Update the relevant guide whenever arguments, defaults, output locations, or resume behavior change.
- Use placeholder paths in reusable examples. Identify literal site-specific defaults in fixed-script guides and state what users must adjust.
- Link to Unity's official documentation for HPC policies, partition/GPU inventories, access, modules, and configuration rather than maintaining copies of those changing details here.
