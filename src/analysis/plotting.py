import matplotlib.pyplot as plt
import numpy as np
import torch


def to_01(img):
    """Convert tensor/array to numpy float in [0, 1]."""
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    img = img.astype(np.float64)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    return img


def plot_rotation_heatmaps(rotation_results, save_path=None):
    """rotation_results: dict {angle: {"corr": ..., ...}} as returned by
    analysis.correlation.compute_all_rotations."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    items = list(rotation_results.items())
    vmin = min(r["corr"].min() for _, r in items)
    vmax = max(r["corr"].max() for _, r in items)

    im = None
    for ax, (angle, r) in zip(axes.flat, items):
        im = ax.imshow(r["corr"], cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(f"Train rotated {angle}°")
        ax.set_xlabel("Horizontal shift")
        ax.set_ylabel("Vertical shift")

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.92, label="Max |Pearson correlation|")

    if save_path:
        fig.savefig(save_path)
    return fig


def plot_correlation_histogram(rotation_results, save_path=None):
    all_vals = np.stack([r["corr"] for r in rotation_results.values()], axis=0).ravel()

    fig = plt.figure(figsize=(8, 4))
    plt.hist(all_vals, bins=50, color="teal", edgecolor="black", alpha=0.7)
    plt.xlabel("Max |Pearson correlation|")
    plt.ylabel("Frequency")
    plt.title("Distribution of best correlation values (all rotations & shifts)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    print(
        f"Correlation stats: min={all_vals.min():.3f}, max={all_vals.max():.3f}, "
        f"mean={all_vals.mean():.3f}, median={np.median(all_vals):.3f}"
    )

    if save_path:
        fig.savefig(save_path)
    return fig
