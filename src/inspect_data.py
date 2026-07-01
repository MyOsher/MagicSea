"""Week 1 sanity check — print stats and save a PNG preview of processed maps.

    python -m src.inspect_data data/processed/<file>_64x64.npy
"""
from pathlib import Path
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless: save to file, no display needed
import matplotlib.pyplot as plt

from src.config import load_config


def inspect(npy_path: str | Path, cfg: dict) -> None:
    arr = np.load(npy_path)  # (days, size, size)
    print(f"array: {arr.shape}  dtype={arr.dtype}")
    print(f"  min/mean/max: {np.nanmin(arr):.2f} / {np.nanmean(arr):.2f} / {np.nanmax(arr):.2f}")
    print(f"  missing:      {100 * np.isnan(arr).mean():.1f}%")

    n = min(6, arr.shape[0])
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    axes = np.atleast_1d(axes)
    for i in range(n):
        im = axes[i].imshow(arr[i], origin="lower", cmap="viridis")
        axes[i].set_title(f"day {i}")
        axes[i].axis("off")
    fig.colorbar(im, ax=list(axes), shrink=0.7, label="value")

    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (Path(npy_path).stem + "_preview.png")
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print(f"  preview saved: {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m src.inspect_data path/to/processed.npy")
    inspect(sys.argv[1], load_config())
