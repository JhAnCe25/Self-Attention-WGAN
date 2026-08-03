"""CLI entry point for training.

Usage:
    python scripts/train.py --config configs/experiments/att-1.yaml
    python scripts/train.py --config configs/experiments/wnlv6-6.yaml
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data import MatDataset, make_train_val_split
from src.training import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to an experiment config yaml (overrides applied on top of configs/base.yaml)",
    )
    cli_args = parser.parse_args()

    args = load_config(cli_args.config)

    for sub in ["", "samples", "checkpoints", "matlab_data"]:
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)
    print("Output directory ready:", args.output_dir)

    dataset = MatDataset(mat_folder=args.data_root, image_key=args.image_key, crop_size=args.crop_size)
    train_dataset, val_dataset = make_train_val_split(dataset, train_frac=args.train_frac)

    print(f"Total dataset size : {len(dataset)}")
    print(f"Training set size  : {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    train(args, train_dataset)


if __name__ == "__main__":
    main()
