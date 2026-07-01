"""Week 2 deliverable — evaluate the interpolation baseline over a dataset.

For every map it punches artificial holes, fills them with the baseline,
scores RMSE/MAE on the hidden pixels, then saves:
  - output/baseline_metrics.json  (aggregate + per-map numbers)
  - output/baseline_example.png   (truth / holes / filled / error, one example)

This is the reference the U-Net (weeks 3-4) must beat.

    python -m src.run_baseline
    python -m src.run_baseline data/processed/<file>_64x64.npy --kind blob --seed 0
"""
from pathlib import Path
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import load_config
from src.masks import make_hole, apply_mask
from src.baseline import interpolate_fill
from src.metrics import evaluate


def _default_processed(cfg):
    proc = sorted(Path(cfg["paths"]["processed_dir"]).glob("*_64x64.npy"))
    if not proc:
        raise SystemExit("No processed data found. Run the Week 1 pipeline first.")
    return proc[0]


def run(npy_path, cfg, kind="mixed", seed=0):
    arr = np.load(npy_path)  # (days, H, W)
    rng = np.random.default_rng(seed)

    per_map, example = [], None
    for i, truth in enumerate(arr):
        if (~np.isnan(truth)).sum() == 0:
            continue  # skip all-land maps
        hole = make_hole(truth.shape, rng, kind=kind)
        corrupted, eval_mask = apply_mask(truth, hole)
        if eval_mask.sum() == 0:
            continue  # hole landed entirely on land; skip
        filled = interpolate_fill(corrupted)
        m = evaluate(filled, truth, eval_mask)
        m["day"] = i
        per_map.append(m)
        if example is None:
            example = (truth, corrupted, filled, eval_mask)

    rmses = np.array([m["rmse"] for m in per_map])
    maes = np.array([m["mae"] for m in per_map])
    summary = {
        "variable": cfg["variable"]["short_name"],
        "baseline": "interpolation (linear + nearest fallback)",
        "hole_kind": kind,
        "n_maps": len(per_map),
        "rmse_mean": float(rmses.mean()),
        "rmse_std": float(rmses.std()),
        "mae_mean": float(maes.mean()),
        "mae_std": float(maes.std()),
        "per_map": per_map,
    }

    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "baseline_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    _save_example(example, out_dir / "baseline_example.png", cfg)

    print(f"maps scored:  {summary['n_maps']}")
    print(f"baseline RMSE: {summary['rmse_mean']:.3f} +/- {summary['rmse_std']:.3f}")
    print(f"baseline MAE:  {summary['mae_mean']:.3f} +/- {summary['mae_std']:.3f}")
    print(f"saved: {out_dir/'baseline_metrics.json'} , {out_dir/'baseline_example.png'}")
    return summary


def _save_example(example, path, cfg):
    truth, corrupted, filled, eval_mask = example
    unit = "m" if cfg["variable"]["short_name"] == "wave" else ""
    err = np.where(eval_mask, np.abs(filled - truth), np.nan)
    panels = [
        ("truth", truth, "viridis"),
        ("with holes", corrupted, "viridis"),
        ("baseline filled", filled, "viridis"),
        ("abs error", err, "magma"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (title, data, cmap) in zip(axes, panels):
        im = ax.imshow(data, origin="lower", cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.7, label=unit)
    fig.savefig(path, dpi=100, bbox_inches="tight")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("npy_path", nargs="?", default=None)
    p.add_argument("--kind", default="mixed", choices=["rect", "blob", "mixed"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = load_config()
    path = args.npy_path or _default_processed(cfg)
    run(path, cfg, kind=args.kind, seed=args.seed)
