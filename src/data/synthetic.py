"""Week 1 fallback — generate synthetic marine maps as a NetCDF file.

Lets you build and test the ENTIRE pipeline (weeks 2-5) before you have
Copernicus credentials. The output NetCDF has the same shape/variable name
as the real download, so `src/data/load.py` treats both identically.

    python -m src.data.synthetic
"""
from pathlib import Path

import numpy as np
import xarray as xr

from src.config import load_config


def _smooth_field(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """A smooth, ocean-like scalar field via summed low-frequency waves."""
    ys, xs = np.linspace(0, 3 * np.pi, h), np.linspace(0, 3 * np.pi, w)
    gx, gy = np.meshgrid(xs, ys)
    field = np.zeros((h, w), dtype=np.float32)
    for _ in range(6):
        fx, fy = rng.uniform(0.3, 1.5, size=2)
        phase = rng.uniform(0, 2 * np.pi)
        field += rng.uniform(0.5, 2.0) * np.sin(fx * gx + fy * gy + phase)
    # Normalize to a realistic SST-in-Celsius range (~16-26 C).
    field = (field - field.min()) / (np.ptp(field) + 1e-9)
    return (16.0 + 10.0 * field).astype(np.float32)


def generate(cfg: dict, n_days: int = 14, seed: int = 0) -> Path:
    region, t, var = cfg["region"], cfg["time"], cfg["variable"]
    # Use a coarse native grid; load.py downsamples to target_size anyway.
    h = w = 128
    rng = np.random.default_rng(seed)

    base = _smooth_field(h, w, rng)
    days = []
    for d in range(n_days):
        # Gentle day-to-day drift so the "time" dimension is meaningful.
        days.append(base + 0.15 * d + rng.normal(0, 0.05, size=(h, w)).astype(np.float32))
    data = np.stack(days, axis=0)  # (time, lat, lon), degrees Celsius

    lats = np.linspace(region["lat_min"], region["lat_max"], h, dtype=np.float32)
    lons = np.linspace(region["lon_min"], region["lon_max"], w, dtype=np.float32)
    times = np.array(np.datetime64(t["start"]) + np.arange(n_days), dtype="datetime64[D]")

    ds = xr.Dataset(
        {var["copernicus_var"]: (("time", "latitude", "longitude"), data)},
        coords={"time": times, "latitude": lats, "longitude": lons},
        attrs={"source": "SYNTHETIC — not real observations", "units": "degC"},
    )

    raw_dir = Path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{region['name']}_{var['short_name']}_SYNTHETIC.nc"
    ds.to_netcdf(path)
    print(f"Wrote synthetic dataset: {path}  shape={data.shape}")
    return path


if __name__ == "__main__":
    # Synthetic data is already in Celsius; load.py auto-detects and won't
    # re-offset it, so no config changes are needed.
    generate(load_config())
