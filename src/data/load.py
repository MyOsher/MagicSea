"""Week 1 goal — turn a NetCDF file into a clean NumPy array ready for AI.

    python -m src.data.load path/to/file.nc

Steps:
  1. Open the NetCDF with xarray.
  2. Pick the variable of interest.
  3. Convert units (Kelvin -> Celsius) if configured.
  4. Downsample every daily map to target_size x target_size (the key speed tip).
  5. Save one stacked NumPy array of shape (days, size, size) to data/processed/.

Land / missing pixels stay as np.nan on purpose — that is the real-world
"holes" signal the model will learn to respect.
"""
from pathlib import Path
import sys
import warnings

import numpy as np
import xarray as xr

from src.config import load_config


def _downsample(arr: np.ndarray, size: int) -> np.ndarray:
    """Block-average a 2D map down to (size, size), NaN-aware."""
    h, w = arr.shape
    # Trim so dimensions divide evenly into `size` blocks.
    bh, bw = h // size, w // size
    if bh == 0 or bw == 0:
        raise ValueError(f"Map {h}x{w} is smaller than target {size}x{size}.")
    arr = arr[: bh * size, : bw * size]
    blocks = arr.reshape(size, bh, size, bw)
    # nanmean ignores holes; an all-NaN block correctly stays NaN. Suppress the
    # expected "Mean of empty slice" warning for all-land blocks.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(blocks, axis=(1, 3)).astype(np.float32)


def load(nc_path: str | Path, cfg: dict) -> np.ndarray:
    var = cfg["variable"]
    pp = cfg["preprocess"]
    ds = xr.open_dataset(nc_path)

    name = var["copernicus_var"]
    if name not in ds:
        raise KeyError(f"Variable '{name}' not in file. Found: {list(ds.data_vars)}")

    values = ds[name].values  # (time, lat, lon) or (lat, lon)
    if values.ndim == 2:
        values = values[None, ...]

    # Auto-convert Kelvin -> Celsius. Real Copernicus SST is in Kelvin (~290),
    # our synthetic data is already in Celsius (~20). Detect by magnitude so we
    # never depend on a manual config toggle.
    if pp.get("auto_celsius", True) and np.nanmedian(values) > 100:
        values = values - 273.15
        print("  units:   detected Kelvin -> converted to Celsius")

    size = pp["target_size"]
    stacked = np.stack([_downsample(day, size) for day in values], axis=0)

    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    out = processed_dir / (Path(nc_path).stem + f"_{size}x{size}.npy")
    np.save(out, stacked)

    n_nan = np.isnan(stacked).sum()
    print(f"Loaded {nc_path}")
    print(f"  shape:   {stacked.shape}  (days, {size}, {size})")
    print(f"  range:   {np.nanmin(stacked):.2f} .. {np.nanmax(stacked):.2f}")
    print(f"  missing: {n_nan} pixels ({100 * n_nan / stacked.size:.1f}%)")
    print(f"  saved:   {out}")
    return stacked


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m src.data.load path/to/file.nc")
    load(sys.argv[1], load_config())
