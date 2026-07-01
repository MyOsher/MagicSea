"""Week 1 fallback — generate synthetic marine maps as a NetCDF file.

Lets you build and test the ENTIRE pipeline (weeks 2-5) before you have
Copernicus credentials. The output NetCDF has the same shape/variable name
as the real download, so `src/data/load.py` treats both identically.

Produces a wave-height-like field (VHM0, meters) with a land mask of NaNs,
mirroring how real Copernicus wave data has no values over land.

    python -m src.data.synthetic
"""
from pathlib import Path

import numpy as np
import xarray as xr

from src.config import load_config


def _smooth_field(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """A smooth, ocean-like scalar field in [0, 1] via low-frequency waves."""
    ys, xs = np.linspace(0, 3 * np.pi, h), np.linspace(0, 3 * np.pi, w)
    gx, gy = np.meshgrid(xs, ys)
    field = np.zeros((h, w), dtype=np.float32)
    for _ in range(6):
        fx, fy = rng.uniform(0.3, 1.5, size=2)
        phase = rng.uniform(0, 2 * np.pi)
        field += rng.uniform(0.5, 2.0) * np.sin(fx * gx + fy * gy + phase)
    return ((field - field.min()) / (np.ptp(field) + 1e-9)).astype(np.float32)


def _land_mask(h: int, w: int) -> np.ndarray:
    """A static 'coastline' — top-left corner is land (True = land = NaN)."""
    ys, xs = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    # Land where we're near the top-left corner; a soft diagonal coastline.
    return (xs + ys) < 0.35


def generate(cfg: dict, n_days: int = 14, seed: int = 0, scene_len: int = 7) -> Path:
    region, t, var = cfg["region"], cfg["time"], cfg["variable"]
    # Use a coarse native grid; load.py downsamples to target_size anyway.
    h = w = 128
    rng = np.random.default_rng(seed)
    land = _land_mask(h, w)

    # Draw a fresh base "weather pattern" every `scene_len` days so the maps
    # are genuinely diverse (something for a model to learn), while days within
    # a scene stay temporally continuous.
    days = []
    base = _smooth_field(h, w, rng)
    for d in range(n_days):
        if d % scene_len == 0:
            base = _smooth_field(h, w, rng)
            amp = rng.uniform(3.0, 6.0)      # scene-specific wave energy
            floor = rng.uniform(0.2, 1.0)
        day = floor + amp * base + 0.05 * (d % scene_len) \
            + rng.normal(0, 0.05, size=(h, w)).astype(np.float32)
        day[land] = np.nan  # no waves on land
        days.append(day.astype(np.float32))
    data = np.stack(days, axis=0)  # (time, lat, lon), meters

    lats = np.linspace(region["lat_min"], region["lat_max"], h, dtype=np.float32)
    lons = np.linspace(region["lon_min"], region["lon_max"], w, dtype=np.float32)
    times = np.array(np.datetime64(t["start"]) + np.arange(n_days), dtype="datetime64[D]")

    ds = xr.Dataset(
        {var["copernicus_var"]: (("time", "latitude", "longitude"), data)},
        coords={"time": times, "latitude": lats, "longitude": lons},
        attrs={"source": "SYNTHETIC — not real observations", "units": "m"},
    )

    raw_dir = Path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{region['name']}_{var['short_name']}_SYNTHETIC.nc"
    ds.to_netcdf(path)
    print(f"Wrote synthetic dataset: {path}  shape={data.shape}  "
          f"(land NaN: {100 * np.isnan(data).mean():.0f}%)")
    return path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=14,
                   help="number of daily maps to generate (use more for training)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    generate(load_config(), n_days=args.days, seed=args.seed)
