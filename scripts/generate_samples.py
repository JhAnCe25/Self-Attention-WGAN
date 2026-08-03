"""Generate samples from a trained checkpoint and export to Octave .mat files.

Usage:
    python scripts/generate_samples.py \
        --output-dir /local/janccoce/WGANProject/Att-1 \
        --epoch 25 --resolution 512 --sample-size 10
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.octave_io import generate_and_export

RESOLUTION_TO_NOISE = {
    "128": lambda base: base,
    "256": lambda base: base * 2,
    "512": lambda base: base * 4,
    "1024": lambda base: base * 8,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epoch", type=int, default=None, help="Defaults to latest checkpoint")
    parser.add_argument("--resolution", choices=list(RESOLUTION_TO_NOISE), default="512")
    parser.add_argument("--noise-channels", type=int, default=2)
    parser.add_argument("--channels-img", type=int, default=2)
    parser.add_argument("--base-noise-size", type=int, default=30, help="noise_height/width + pad used in training")
    parser.add_argument("--sample-size", type=int, default=10)
    args = parser.parse_args()

    noise_size = RESOLUTION_TO_NOISE[args.resolution](args.base_noise_size)

    generate_and_export(
        output_dir=args.output_dir,
        noise_channels=args.noise_channels,
        noise_height=noise_size,
        noise_width=noise_size,
        channels_img=args.channels_img,
        sample_size=args.sample_size,
        epoch=args.epoch,
        gen_model=None,  # forces loading from checkpoint
        output_resolution=args.resolution,
    )


if __name__ == "__main__":
    main()
