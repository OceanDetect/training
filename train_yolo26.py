"""Reusable YOLO26 training: python train_yolo26.py --data /path/to/data.yaml."""

import argparse
import json
from pathlib import Path
import sys


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return number


def probability(value):
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def positive_float(value):
    number = float(value)
    if not 0 < number < float("inf"):
        raise argparse.ArgumentTypeError("must be a finite number greater than 0")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train YOLO26 using a dataset YAML and explicit run settings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    paths = parser.add_argument_group("Dataset and output")
    paths.add_argument("--data", type=Path, required=True, help="Detection dataset YAML")
    paths.add_argument("--model", default="yolo26l.pt", help="Model weights filename or path")
    paths.add_argument("--project", type=Path, default=Path("runs"), help="Output root")
    paths.add_argument("--name", help="Run name; defaults to model, augmentation, and image size")

    training = parser.add_argument_group("Training")
    training.add_argument("--epochs", type=positive_int, default=200, help="Maximum epochs")
    training.add_argument("--imgsz", type=positive_int, default=1280, help="Square image size")
    training.add_argument("--patience", type=nonnegative_int, default=20,
                          help="Early-stopping patience in epochs; 0 disables early stopping")
    training.add_argument("--batch", type=positive_int, default=16, help="Images per batch")
    training.add_argument("--workers", type=nonnegative_int, default=8, help="Data loader workers")
    training.add_argument("--seed", type=nonnegative_int, default=42, help="Random seed")
    training.add_argument("--optimizer", choices=("AdamW", "SGD", "auto"), default="AdamW",
                          help="Optimizer; auto lets Ultralytics choose optimizer and learning rate")
    training.add_argument("--lr0", type=positive_float, default=0.001, help="Initial learning rate")
    training.add_argument("--weight-decay", type=probability, default=0.0005, help="Weight decay")
    training.add_argument("--warmup-epochs", type=nonnegative_int, default=3, help="Warmup epochs")
    training.add_argument("--fraction", type=positive_float, default=1.0,
                          help="Fraction of training images to use, in (0, 1]; validation is unchanged")
    training.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    training.add_argument("--no-deterministic", action="store_true", help="Disable deterministic operations")

    augmentation = parser.add_argument_group("Augmentation")
    augmentation.add_argument("--augmentation", choices=("mixed", "standard"), default="mixed",
                              help="Mixed adds custom grayscale/blur/CLAHE; standard uses YOLO defaults")
    augmentation.add_argument("--grayscale", type=probability, default=0.20, help="Grayscale probability (mixed)")
    augmentation.add_argument("--blur", type=probability, default=0.10, help="Mild blur probability (mixed)")
    augmentation.add_argument("--clahe", type=probability, default=0.10, help="Local contrast probability (mixed)")
    augmentation.add_argument("--mosaic", type=probability, default=0.5, help="Mosaic probability")
    augmentation.add_argument("--close-mosaic", type=nonnegative_int, default=10,
                              help="Disable mosaic in final N epochs; use 0 for a short smoke run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print resolved settings and exit without dependencies, GPU, or file writes")
    args = parser.parse_args(argv)
    if args.fraction > 1:
        parser.error("--fraction must be in (0, 1]")
    return args


def training_settings(args):
    name = args.name or f"{Path(args.model).stem}_{args.augmentation}_{args.imgsz}"
    return {
        "data": str(args.data.expanduser().resolve()),
        "project": str(args.project.expanduser().resolve()),
        "name": name,
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "seed": args.seed,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "fraction": args.fraction,
        "mosaic": args.mosaic,
        "close_mosaic": args.close_mosaic,
        "amp": not args.no_amp,
        "deterministic": not args.no_deterministic,
        "cache": False,
        "device": 0,  # First GPU visible inside the SLURM allocation.
        "exist_ok": False,
    }


def main(argv=None):
    args = parse_args(argv)
    settings = training_settings(args)
    print("OceanDetect YOLO26 training")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Model: {args.model}")
    print(f"Augmentation: {args.augmentation}")
    if args.augmentation == "mixed":
        print(f"Custom probabilities: grayscale={args.grayscale}, blur={args.blur}, CLAHE={args.clahe}")
    print(json.dumps(settings, indent=2))
    if args.dry_run:
        print("Dry run complete; no training or environment validation performed.")
        return

    if not Path(settings["data"]).is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {settings['data']}")

    # Import only after parsing so --help and --dry-run work without GPU packages.
    import torch
    import ultralytics
    from ultralytics import YOLO

    print(f"Ultralytics: {ultralytics.__version__}")
    print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the GPU allocation and Python environment.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.augmentation == "mixed":
        import albumentations as A
        from ultralytics.cfg import get_cfg
        from ultralytics.data.augment import Albumentations

        print(f"Albumentations: {A.__version__}")
        transforms = [
            A.ToGray(num_output_channels=3, p=args.grayscale),
            A.Blur(blur_limit=(3, 5), p=args.blur),
            A.CLAHE(clip_limit=(1.0, 2.0), p=args.clahe),
        ]
        get_cfg(overrides={"augmentations": transforms})
        if Albumentations(p=1.0, transforms=transforms).transform is None:
            raise RuntimeError("Custom augmentation initialization failed. Check dependency versions.")
        for transform in transforms:
            print(f"  {transform}")
        settings["augmentations"] = transforms

    model = YOLO(args.model)
    model.train(**settings)
    print(f"Training complete. Results saved to: {model.trainer.save_dir}")


if __name__ == "__main__":
    main()
