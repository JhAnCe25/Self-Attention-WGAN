"""Check whether the generator is memorizing training textures.

Generates one sample, then compares it (across a shift grid and 4 rotations)
against every image in the validation set, looking for suspiciously high
Pearson correlation with any single training/validation image.

Usage:
    python scripts/run_overfit_check.py \
        --checkpoint /local/janccoce/WGANProject/outputWNLV6-10/checkpoints/generator_epoch_25.pth \
        --data-root /local/janccoce/WGANProject/dataV6norm/ \
        --step 2
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis import (
    compute_all_rotations,
    plot_correlation_histogram,
    plot_rotation_heatmaps,
)
from src.data import MatDataset, make_train_val_split
from src.models import Generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to generator_epoch_N.pth")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--noise-channels", type=int, default=2)
    parser.add_argument("--noise-height", type=int, default=30)
    parser.add_argument("--noise-width", type=int, default=30)
    parser.add_argument("--channels-img", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--step", type=int, default=2, help="Shift step for the correlation grid")
    parser.add_argument("--out-dir", default="overfit_check_out")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load generator and produce one sample
    netG = Generator(in_channels=args.noise_channels, out_channels=args.channels_img).to(device)
    netG.load_state_dict(torch.load(args.checkpoint, map_location=device))
    netG.eval()

    noise = torch.randn(args.batch_size, args.noise_channels, args.noise_height, args.noise_width, device=device)
    with torch.no_grad():
        fake_imgs = netG(noise)

    gen_full = fake_imgs[0].cpu().numpy()
    gen_ch1 = gen_full[0]

    # 2. Load validation images (same channel convention as training)
    dataset = MatDataset(mat_folder=args.data_root, image_key="datatmp")
    _, val_dataset = make_train_val_split(dataset)
    dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    train_imgs = []
    for batch in dataloader:
        train_imgs.append(batch[:, 0, :, :])
    train_imgs = torch.cat(train_imgs, dim=0).numpy()

    # 3. Compute correlation across shifts and rotations
    rotation_results = compute_all_rotations(gen_ch1, train_imgs, device, step=args.step)

    plot_rotation_heatmaps(rotation_results, save_path=os.path.join(args.out_dir, "rotation_heatmaps.png"))
    plot_correlation_histogram(rotation_results, save_path=os.path.join(args.out_dir, "correlation_histogram.png"))

    print(f"Saved plots to {args.out_dir}/")


if __name__ == "__main__":
    main()
