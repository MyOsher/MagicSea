"""Strengthen the proof — RMSE vs hole size, U-Net against a panel of baselines.

The compelling POC story isn't a single number: it's that the U-Net's advantage
*holds or grows* as gaps get larger, where simple methods degrade fastest.

For each hole-coverage level we punch blob holes of that size into every held-out
map and score four methods on the hidden pixels (original units, meters):
  mean fill  |  nearest neighbour  |  linear interpolation  |  U-Net

    python -m src.robustness

Saves:
  output/robustness.json                  full table (method x coverage)
  output/robustness_rmse_vs_holesize.png  the money chart
"""
from pathlib import Path
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import load_config
from src.model import UNet
from src.masks import random_blob, apply_mask
from src.baseline import mean_fill, nearest_fill, interpolate_fill
from src.metrics import rmse
from src.train import split_maps

COVERAGES = [0.05, 0.10, 0.20, 0.35, 0.50]
CLASSICAL = {"mean": mean_fill, "nearest": nearest_fill, "interpolation": interpolate_fill}


def _unet_fill(truth, hole, ocean, net, mean, std, device):
    """Run the U-Net on one map/hole and return a filled map in original units."""
    norm = (truth - mean) / std
    known = ocean & ~hole
    filled = np.where(known, norm, 0.0).astype(np.float32)
    inp = np.stack([filled, known.astype(np.float32)], axis=0)
    with torch.no_grad():
        out = net(torch.from_numpy(inp)[None].to(device))[0, 0].cpu().numpy()
    return out * std + mean


def run(cfg, npy_path, seed=0, split="random"):
    out_dir = Path(cfg["paths"]["output_dir"])
    with open(out_dir / "norm_stats.json") as f:
        stats = json.load(f)
    mean, std = stats["mean"], stats["std"]

    maps = np.load(npy_path)
    _, val_maps = split_maps(maps, seed=seed, mode=split)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = UNet().to(device)
    net.load_state_dict(torch.load(out_dir / "unet.pt", map_location=device))
    net.eval()

    methods = list(CLASSICAL) + ["unet"]
    # results[method][coverage] = list of per-map RMSE
    results = {m: {c: [] for c in COVERAGES} for m in methods}
    actual_cov = {c: [] for c in COVERAGES}

    for cov in COVERAGES:
        # Fresh RNG per level so every method sees identical holes at this size.
        for i, truth in enumerate(val_maps):
            ocean = ~np.isnan(truth)
            if ocean.sum() == 0:
                continue
            rng = np.random.default_rng(1000 * int(cov * 100) + i)
            hole = random_blob(truth.shape, rng, coverage=cov) & ocean
            if hole.sum() == 0:
                continue
            actual_cov[cov].append(hole.sum() / ocean.sum())
            corrupted, _ = apply_mask(truth, hole)

            for name, fn in CLASSICAL.items():
                results[name][cov].append(rmse(fn(corrupted), truth, hole))
            uf = _unet_fill(truth, hole, ocean, net, mean, std, device)
            results["unet"][cov].append(rmse(uf, truth, hole))

    summary = {"coverages": COVERAGES,
               "actual_coverage_mean": {c: float(np.mean(actual_cov[c])) for c in COVERAGES},
               "rmse_mean": {m: {c: float(np.mean(results[m][c])) for c in COVERAGES}
                             for m in methods}}
    with open(out_dir / "robustness.json", "w") as f:
        json.dump(summary, f, indent=2)

    _plot(summary, methods, out_dir / "robustness_rmse_vs_holesize.png")
    _print_table(summary, methods)
    return summary


def _print_table(summary, methods):
    covs = summary["coverages"]
    header = "coverage  " + "".join(f"{m:>16}" for m in methods)
    print(header)
    for c in covs:
        row = f"{c*100:6.0f}%  " + "".join(
            f"{summary['rmse_mean'][m][c]:>16.3f}" for m in methods)
        print(row)
    best = summary["rmse_mean"]
    big = covs[-1]
    imp = 100 * (best["interpolation"][big] - best["unet"][big]) / best["interpolation"][big]
    print(f"\nAt {big*100:.0f}% holes: U-Net beats interpolation by {imp:+.1f}% RMSE.")


def _plot(summary, methods, path):
    covs = [c * 100 for c in summary["coverages"]]
    plt.figure(figsize=(7, 5))
    for m in methods:
        ys = [summary["rmse_mean"][m][c] for c in summary["coverages"]]
        style = dict(lw=2.5, marker="o") if m == "unet" else dict(lw=1.5, marker=".", ls="--")
        plt.plot(covs, ys, label=m, **style)
    plt.xlabel("hole coverage (% of ocean hidden)")
    plt.ylabel("RMSE on hidden pixels (m)")
    plt.title("Gap-filling error vs hole size")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(path, dpi=100, bbox_inches="tight")


if __name__ == "__main__":
    cfg = load_config()
    path = sorted(Path(cfg["paths"]["processed_dir"]).glob("*_64x64.npy"))[0]
    run(cfg, path)
