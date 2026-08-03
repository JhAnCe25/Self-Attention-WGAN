import os

import numpy as np
import torch

from .models import Generator


def save_octave_matrix(arr, path, variable_name="datatmp"):
    """
    Saves a 3D NumPy array (H, W, C) to an Octave-compatible .mat text file.
    Assumes real-valued data and saves it in complex format (real_part, 0.0),
    matching the format the MATLAB/Octave multifractal analysis pipeline expects.
    """
    if arr.ndim != 3:
        raise ValueError(f"Input array must be 3D (H, W, C), but got {arr.ndim}D.")

    H, W, C = arr.shape

    with open(path, "w") as f:
        f.write("# Created by Python script\n")
        f.write(f"# name: {variable_name}\n")
        f.write("# type: complex matrix\n")
        f.write("# ndims: 3\n")
        f.write(f" {H} {W} {C}\n")

        for k in range(C):
            for j in range(W):
                for i in range(H):
                    val = arr[i, j, k]
                    f.write(f"({val},0.0)\n")


def get_latest_epoch(checkpoint_dir):
    """Return the highest epoch number found in *checkpoint_dir*, or -1."""
    ckpts = [
        f
        for f in os.listdir(checkpoint_dir)
        if f.startswith("generator_epoch_") and f.endswith(".pth")
    ]
    if not ckpts:
        return -1
    epochs = [int(f.split("_")[-1].split(".")[0]) for f in ckpts]
    return max(epochs)


def generate_and_export(
    output_dir,
    noise_channels,
    noise_height,
    noise_width,
    channels_img,
    sample_size=20,
    epoch=None,
    gen_model=None,
    output_resolution=None,
):
    """
    Export synthetic textures to Octave .mat files.

    The generator's output size = noise_size * 8 (three x2 upsampling stages).
    Example: noise (64x64) -> output (512x512), noise (128x128) -> output (1024x1024).

    Args:
        output_resolution: if provided, used in folder naming (e.g. "512" or "1024").
                            If None, auto-detected as noise_height*8.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if gen_model is None:
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        if not os.path.isdir(checkpoint_dir):
            print(f"[ERROR] Checkpoint directory not found: {checkpoint_dir}")
            return
        if epoch is None:
            epoch = get_latest_epoch(checkpoint_dir)
            if epoch < 0:
                print("[ERROR] No generator checkpoints found to load.")
                return
        ckpt_path = os.path.join(checkpoint_dir, f"generator_epoch_{epoch}.pth")
        if not os.path.isfile(ckpt_path):
            print(f"[ERROR] Checkpoint not found: {ckpt_path}")
            return
        print(f"Loading checkpoint: {ckpt_path}")
        gen = Generator(in_channels=noise_channels, out_channels=channels_img).to(device)
        gen.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        gen = gen_model

    out_height = noise_height * 8
    out_width = noise_width * 8
    output_resolution = (
        f"{out_height}x{out_width}" if output_resolution is None else str(output_resolution)
    )

    gen.eval()
    with torch.no_grad():
        noise = torch.randn(sample_size, noise_channels, noise_height, noise_width, device=device)
        fake_samples = gen(noise).cpu().numpy()  # (sample_size, C, out_h, out_w)

    print(f"Generated tensor shape: {fake_samples.shape} (output resolution {output_resolution})")

    generation_sub_dir = (
        f"epoch_{epoch:03d}_{output_resolution}" if epoch is not None else f"final_generation_{output_resolution}"
    )
    out_dir = os.path.join(output_dir, "matlab_data", generation_sub_dir)
    os.makedirs(out_dir, exist_ok=True)

    file_path = None
    for i in range(sample_size):
        single_sample = fake_samples[i]
        single_sample_hwc = np.transpose(single_sample, (1, 2, 0))
        filename = f"MRW2DGEN{1001 + i:04d}.mat"
        file_path = os.path.join(out_dir, filename)
        save_octave_matrix(single_sample_hwc, file_path, variable_name="datatmp")
    print(f"Saved Octave .mat  -> {file_path}")
