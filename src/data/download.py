"""Week 1 — download real marine data from Copernicus Marine Service.

Usage (once you have credentials in .env):
    pip install -r requirements.txt
    python -m src.data.download

This uses the official `copernicusmarine` toolbox. If you don't have
credentials yet, use `src/data/synthetic.py` instead to keep moving.
"""
from pathlib import Path
import os

from dotenv import load_dotenv

from src.config import load_config


def download(cfg: dict) -> Path:
    # Import here so the rest of the project runs without the toolbox installed.
    import copernicusmarine

    load_dotenv()  # read COPERNICUSMARINE_SERVICE_USERNAME / _PASSWORD from .env
    if not os.getenv("COPERNICUSMARINE_SERVICE_USERNAME"):
        raise SystemExit(
            "No Copernicus credentials found.\n"
            "  1. Register (free): https://data.marine.copernicus.eu/register\n"
            "  2. cp .env.example .env  and fill in your username/password\n"
            "  3. Or skip this and run:  python -m src.data.synthetic"
        )

    region, t, var = cfg["region"], cfg["time"], cfg["variable"]
    raw_dir = Path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{region['name']}_{var['short_name']}_{t['start']}_{t['end']}.nc"

    print(f"Downloading {var['copernicus_var']} for {region['name']} "
          f"({t['start']} -> {t['end']})...")
    copernicusmarine.subset(
        dataset_id=var["copernicus_dataset_id"],
        variables=[var["copernicus_var"]],
        minimum_longitude=region["lon_min"],
        maximum_longitude=region["lon_max"],
        minimum_latitude=region["lat_min"],
        maximum_latitude=region["lat_max"],
        start_datetime=f"{t['start']}T00:00:00",
        end_datetime=f"{t['end']}T00:00:00",
        output_directory=str(raw_dir),
        output_filename=out_name,
    )
    path = raw_dir / out_name
    print(f"Saved: {path}")
    return path


if __name__ == "__main__":
    download(load_config())
