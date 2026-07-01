"""Week 5 — compare the U-Net against the interpolation baseline.

Both methods see the SAME held-out maps and the SAME holes, so the comparison
is fair. Scores RMSE/MAE on the hidden hole pixels (original units, meters).

    python -m src.evaluate

Saves:
  output/evaluation.json        aggregate + per-map, both methods
  output/evaluation_example.png truth / holes / baseline / U-Net / errors
"""
from pathlib import Path
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import load_config
from src.dataset import WaveInpaintingDataset
from src.model import UNet
from src.baseline import interpolate_fill
from src.metrics import evaluate as score
from src.train import split_maps


def _load_model(out_dir, device):
    net = UNet().to(device)
    net.load_state_dict(torch.load(out_dir / "unet.pt", map_location=device))
    net.eval()
    return net


def run(cfg, npy_path, seed=0, split="random"):
    out_dir = Path(cfg["paths"]["output_dir"])
    with open(out_dir / "norm_stats.json") as f:
        stats = json.load(f)
    mean, std = stats["mean"], stats["std"]

    maps = np.load(npy_path)
    _, val_maps = split_maps(maps, seed=seed, mode=split)  # same held-out split as training
    ds = WaveInpaintingDataset(val_maps, mean, std, augment=False)  # fixed holes

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = _load_model(out_dir, device)

    rows, example = [], None
    for i in range(len(ds)):
        s = ds[i]
        truth = val_maps[i]
        hole = s["hole"][0].numpy().astype(bool)
        if hole.sum() == 0:
            continue

        # Baseline: interpolate the same corrupted map (original units).
        corrupted = truth.copy()
        corrupted[hole] = np.nan
        base_fill = interpolate_fill(corrupted)

        # U-Net: predict in normalized space, then denormalize.
        with torch.no_grad():
            out = net(s["input"][None].to(device))[0, 0].cpu().numpy()
        unet_fill = out * std + mean

        b = score(base_fill, truth, hole)
        u = score(unet_fill, truth, hole)
        rows.append({"map": i, "n_pixels": b["n_pixels"],
                     "baseline_rmse": b["rmse"], "unet_rmse": u["rmse"],
                     "baseline_mae": b["mae"], "unet_mae": u["mae"]})
        if example is None:
            example = (truth, corrupted, base_fill, unet_fill, hole)

    def _mean(key):
        return float(np.mean([r[key] for r in rows]))

    base_rmse, unet_rmse = _mean("baseline_rmse"), _mean("unet_rmse")
    improvement = 100 * (base_rmse - unet_rmse) / base_rmse if base_rmse else 0.0
    summary = {
        "variable": cfg["variable"]["short_name"],
        "n_maps": len(rows),
        "baseline_rmse_mean": base_rmse,
        "unet_rmse_mean": unet_rmse,
        "baseline_mae_mean": _mean("baseline_mae"),
        "unet_mae_mean": _mean("unet_mae"),
        "rmse_improvement_pct": improvement,
        "per_map": rows,
    }
    with open(out_dir / "evaluation.json", "w") as f:
        json.dump(summary, f, indent=2)
    _save_example(example, out_dir / "evaluation_example.png", cfg)

    print(f"held-out maps:  {summary['n_maps']}")
    print(f"baseline RMSE:  {base_rmse:.3f} m")
    print(f"U-Net RMSE:     {unet_rmse:.3f} m")
    print(f"improvement:    {improvement:+.1f}%  (positive = U-Net better)")
    print(f"saved: {out_dir/'evaluation.json'} , {out_dir/'evaluation_example.png'}")
    return summary


def _save_example(example, path, cfg):
    truth, corrupted, base_fill, unet_fill, hole = example
    be = np.where(hole, np.abs(base_fill - truth), np.nan)
    ue = np.where(hole, np.abs(unet_fill - truth), np.nan)
    vmax = float(np.nanmax(truth))
    emax = float(np.nanmax([np.nanmax(be), np.nanmax(ue)]))  # shared error scale
    panels = [
        ("truth", truth, "viridis", 0, vmax),
        ("with holes", corrupted, "viridis", 0, vmax),
        ("baseline", base_fill, "viridis", 0, vmax),
        ("U-Net", unet_fill, "viridis", 0, vmax),
        ("|err| baseline", be, "magma", 0, emax),
        ("|err| U-Net", ue, "magma", 0, emax),
    ]
    fig, axes = plt.subplots(1, 6, figsize=(22, 4))
    for ax, (title, data, cmap, vmin, vmx) in zip(axes, panels):
        im = ax.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmx)
        ax.set_title(title); ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.7, label="m")
    fig.savefig(path, dpi=100, bbox_inches="tight")


if __name__ == "__main__":
    cfg = load_config()
    path = sorted(Path(cfg["paths"]["processed_dir"]).glob("*_64x64.npy"))[0]
    run(cfg, path)
