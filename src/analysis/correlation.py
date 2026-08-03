"""Shift/rotation Pearson-correlation overfitting check.

Ported from the exploratory notebook cells that compared a generated sample
against every training image across a grid of shifts and 4 rotations, to
check whether the generator was memorizing training textures. The logic is
unchanged; it's just expressed as functions taking explicit arguments
instead of relying on notebook execution order and stray globals.
"""
import numpy as np
import torch
from tqdm import tqdm


def compute_corr_matrix(gen_ch1, train_flat_list, device, step=2):
    """
    gen_ch1: numpy array (H, W)
    train_flat_list: list of flattened training images (each length H*W)
    step: shift step (default 2)

    Returns:
      corr_mat: (n_shifts, n_shifts) max |correlation|
      best_idx_mat: (n_shifts, n_shifts) index of best-matching training image
      shifts: the shift values used along each axis
    """
    H, W = gen_ch1.shape
    shifts = np.arange(0, H, step)
    n_shifts = len(shifts)

    corr_mat = np.zeros((n_shifts, n_shifts))
    best_idx_mat = np.zeros((n_shifts, n_shifts), dtype=int)

    train_array = np.array(train_flat_list)
    train_tensor = torch.from_numpy(train_array).float().to(device)
    train_mean = train_tensor.mean(dim=1, keepdim=True)
    train_std = train_tensor.std(dim=1, keepdim=True, unbiased=False)
    denom_train = train_tensor.shape[1] * train_std.squeeze()

    total = n_shifts * n_shifts
    with tqdm(total=total, desc="Full-image GPU correlation") as pbar:
        for i, dy in enumerate(shifts):
            rolled_y = np.roll(gen_ch1, shift=dy, axis=0)
            for j, dx in enumerate(shifts):
                rolled = np.roll(rolled_y, shift=dx, axis=1)
                patch_flat = rolled.ravel()

                patch_tensor = torch.from_numpy(patch_flat).float().to(device)
                patch_mean = patch_tensor.mean()
                patch_std = patch_tensor.std(unbiased=False)

                denom = denom_train * patch_std
                denom = torch.where(denom > 1e-8, denom, torch.full_like(denom, float("nan")))

                corr = ((train_tensor - train_mean) * (patch_tensor - patch_mean)).sum(dim=1) / denom
                corr = torch.nan_to_num(corr, nan=0.0)

                abs_corr = corr.abs()
                best_val, best_idx = torch.max(abs_corr, dim=0)

                corr_mat[i, j] = best_val.item()
                best_idx_mat[i, j] = best_idx.item()
                pbar.update(1)

    return corr_mat, best_idx_mat, shifts


def compute_all_rotations(gen_ch1, train_imgs, device, step=2):
    """Run compute_corr_matrix for the generated image at 0/90/180/270 degree
    rotations of the training set, returning one (corr, best_idx) pair per angle."""
    rotations_k = {0: 0, 90: 1, 180: 2, 270: 3}
    results = {}
    for angle, k in rotations_k.items():
        rotated_train = [np.rot90(img, k=k) for img in train_imgs]
        flat = [img.ravel() for img in rotated_train]
        corr_mat, best_idx_mat, shifts = compute_corr_matrix(gen_ch1, flat, device, step=step)
        results[angle] = {"corr": corr_mat, "best_idx": best_idx_mat, "shifts": shifts}
    return results
