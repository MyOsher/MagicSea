"""Week 2 — a classical gap-filling baseline.

This is the number the neural net must beat. A U-Net that can't beat simple
interpolation isn't worth deploying; one that clearly beats it is the POC's
core evidence.

We fill NaN holes by 2D interpolation over the surrounding valid pixels
(linear, with nearest-neighbour fallback for anything outside the convex hull).

    python -m src.baseline   # self-test on synthetic processed data
"""
import numpy as np
from scipy.interpolate import griddata


def mean_fill(corrupted_map):
    """Naive floor: fill every hole with the mean of the known pixels."""
    filled = corrupted_map.copy()
    valid = ~np.isnan(corrupted_map)
    if valid.sum() == 0:
        return filled
    filled[~valid] = corrupted_map[valid].mean()
    return filled


def nearest_fill(corrupted_map):
    """Fill each hole with its nearest known pixel value."""
    filled = corrupted_map.copy()
    valid = ~np.isnan(corrupted_map)
    if valid.sum() == 0:
        return filled
    h, w = corrupted_map.shape
    yy, xx = np.mgrid[0:h, 0:w]
    pts = np.column_stack([yy[valid], xx[valid]])
    holes = ~valid
    filled[holes] = griddata(pts, corrupted_map[valid],
                             np.column_stack([yy[holes], xx[holes]]),
                             method="nearest")
    return filled


def interpolate_fill(corrupted_map):
    """Fill NaNs in a 2D map by interpolating from valid pixels.

    Returns a copy with every NaN replaced. Pixels that were valid are left
    untouched. If the map is entirely NaN, returns it unchanged.
    """
    filled = corrupted_map.copy()
    valid = ~np.isnan(corrupted_map)
    if valid.sum() == 0:
        return filled

    h, w = corrupted_map.shape
    yy, xx = np.mgrid[0:h, 0:w]
    pts = np.column_stack([yy[valid], xx[valid]])
    vals = corrupted_map[valid]
    holes = ~valid
    query = np.column_stack([yy[holes], xx[holes]])

    # Linear interpolation inside the data hull; nearest fills the rest so no
    # NaN survives at the map edges.
    lin = griddata(pts, vals, query, method="linear")
    nan_after = np.isnan(lin)
    if nan_after.any():
        lin[nan_after] = griddata(pts, vals, query[nan_after], method="nearest")

    filled[holes] = lin
    return filled


if __name__ == "__main__":
    from pathlib import Path
    from src.config import load_config
    from src.masks import make_hole, apply_mask
    from src.metrics import evaluate

    cfg = load_config()
    proc = sorted(Path(cfg["paths"]["processed_dir"]).glob("*_64x64.npy"))
    if not proc:
        raise SystemExit("No processed data. Run the Week 1 pipeline first.")

    arr = np.load(proc[0])
    rng = np.random.default_rng(0)
    truth = arr[0]
    hole = make_hole(truth.shape, rng, kind="blob")
    corrupted, eval_mask = apply_mask(truth, hole)
    filled = interpolate_fill(corrupted)
    print("baseline (interpolation):", evaluate(filled, truth, eval_mask))
