"""One command to run the WHOLE real-data experiment from a NetCDF file.

    python -m src.run_real data/raw/<real_copernicus_file>.nc --epochs 40

It chains the exact same steps the weekly scripts do, but end-to-end and with
its own output folder so REAL results never overwrite the synthetic smoke-test
in output/:

    load .nc -> 64x64 .npy   (src.data.load)
    train U-Net              (src.train)
    evaluate vs baseline     (src.evaluate)
    write output/real/RESULTS_REAL.md   (provenance + real numbers)

Why a separate folder: output/RESULTS.md is explicitly labelled as synthetic.
Real numbers are the deliverable customers can actually be shown, so we keep
them apart and stamp where the data came from.
"""
from pathlib import Path
import argparse
import copy
import json

import numpy as np

from src.config import load_config
from src.data.load import load as load_nc
from src.train import train as train_unet
from src import evaluate as evaluate_mod


def _real_cfg(cfg: dict) -> dict:
    """Clone the config with output redirected to output/real/."""
    real = copy.deepcopy(cfg)
    real["paths"]["output_dir"] = str(Path(cfg["paths"]["output_dir"]) / "real")
    return real


def _write_results_md(cfg: dict, real_cfg: dict, nc_path: Path,
                      npy_path: Path, maps: np.ndarray, summary: dict,
                      epochs: int) -> Path:
    out_dir = Path(real_cfg["paths"]["output_dir"])
    md = out_dir / "RESULTS_REAL.md"
    region, t, var = cfg["region"], cfg["time"], cfg["variable"]
    n_nan = int(np.isnan(maps).sum())
    pct_nan = 100 * n_nan / maps.size if maps.size else 0.0
    base, unet = summary["baseline_rmse_mean"], summary["unet_rmse_mean"]
    imp = summary["rmse_improvement_pct"]

    md.write_text(
        f"""# REAL-DATA Results — Marine Map Gap-Filling (Wave Height)

> These numbers are on **real** data, not the synthetic smoke test. Source and
> preprocessing are stamped below for provenance.

## Data provenance
- **Source file:** `{nc_path.name}`
- **Variable:** `{var['copernicus_var']}` ({var['short_name']}), significant wave height, meters.
- **Dataset id (config):** `{var.get('copernicus_dataset_id', 'n/a')}`
- **Region (config):** lon [{region['lon_min']}, {region['lon_max']}], lat [{region['lat_min']}, {region['lat_max']}] ({region['name']})
- **Time window (config):** {t['start']} -> {t['end']}
- **Processed array:** `{npy_path.name}`  shape `{tuple(int(x) for x in maps.shape)}` (maps, H, W)
- **Value range:** {float(np.nanmin(maps)):.2f} .. {float(np.nanmax(maps)):.2f} m
- **Missing (land/gaps):** {n_nan} pixels ({pct_nan:.1f}%)

## Headline result
| Method | RMSE (m) down | MAE (m) down |
|--------|--------------:|-------------:|
| Interpolation baseline | {base:.3f} | {summary['baseline_mae_mean']:.3f} |
| **U-Net** | **{unet:.3f}** | **{summary['unet_mae_mean']:.3f}** |
| **Improvement** | **{imp:+.1f}%** | |

Scored on {summary['n_maps']} held-out maps, on hidden hole pixels only, land
excluded — the same fair setup as the synthetic run (same maps, same holes for
both methods).

## Artifacts (in output/real/)
- `train_loss.png` — training/validation loss curve.
- `evaluation_example.png` — truth / holes / baseline / U-Net / per-pixel error.
- `evaluation.json` — full per-map numbers, both methods.

## How this was produced
```bash
python -m src.run_real {nc_path.as_posix()} --epochs {epochs}
```
"""
    )
    return md


def main(nc_path: str, epochs: int = 40, batch: int = 8, lr: float = 1e-3, seed: int = 0):
    cfg = load_config()
    real_cfg = _real_cfg(cfg)
    Path(real_cfg["paths"]["output_dir"]).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 1/3  load NetCDF -> 64x64 NumPy")
    print("=" * 60)
    maps = load_nc(nc_path, cfg)
    npy_path = sorted(Path(cfg["paths"]["processed_dir"]).glob("*_64x64.npy"),
                      key=lambda p: p.stat().st_mtime)[-1]
    if len(maps) < 8:
        raise SystemExit(
            f"Only {len(maps)} maps in this file — too few to train/hold-out. "
            "Download a longer time window (months) in config.yaml."
        )

    print("\n" + "=" * 60)
    print(f"STEP 2/3  train U-Net ({epochs} epochs)")
    print("=" * 60)
    train_unet(real_cfg, npy_path, epochs=epochs, batch=batch, lr=lr, seed=seed)

    print("\n" + "=" * 60)
    print("STEP 3/3  evaluate U-Net vs interpolation baseline")
    print("=" * 60)
    summary = evaluate_mod.run(real_cfg, npy_path, seed=seed)

    md = _write_results_md(cfg, real_cfg, Path(nc_path), npy_path, maps, summary, epochs)
    print(f"\nDONE. Real-data report: {md}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("nc_path", help="path to a real NetCDF file (e.g. data/raw/foo.nc)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    ARGS_EPOCHS = args.epochs
    main(args.nc_path, epochs=args.epochs, batch=args.batch, lr=args.lr, seed=args.seed)
