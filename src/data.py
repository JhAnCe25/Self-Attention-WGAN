import os

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


class MatDataset(Dataset):
    """Loads 2-channel complex-valued multifractal textures stored as .mat files."""

    def __init__(self, mat_folder, image_key="datatmp", crop_size=128):
        self.paths = sorted(
            os.path.join(mat_folder, f)
            for f in os.listdir(mat_folder)
            if f.endswith(".mat")
        )
        self.image_key = image_key
        self.crop_size = crop_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        mat_data = loadmat(path)
        arr = mat_data[self.image_key]
        arr = arr.real.astype(np.float32)              # (H, W, C)

        x = torch.from_numpy(arr).permute(2, 0, 1)      # (C, H, W)
        x = x[:, : self.crop_size, : self.crop_size]
        return x


def make_train_val_split(dataset, train_frac=0.8):
    """Ordered (non-shuffled) 80/20 split: first train_frac for train, remainder for val."""
    train_size = int(train_frac * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
    return train_dataset, val_dataset
