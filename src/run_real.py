"""One command to run the WHOLE real-data experiment from a NetCDF file.

    python -m src.run_real data/raw/<real_file>.nc --epochs 40

It chains the same steps the weekly scripts do, but end-to-end, and runs the
evaluation under TWO hold-out schemes so the result is honest:

    load .nc -> .npy                         (src.data.load)
    for split in {temporal, random}:
        train U-Net                          (src.train)
        evaluate vs interpolation baseline   (src.evaluate)
        robustness — RMSE vs hole size       (src.robustness)
    write output/real/RESULTS_REAL.md        (provenance + both splits compared)

Why two splits:
- random   — shuffle then hold out 25% of maps. For an hourly time series the
             held-out frames are near-duplicates of training frames (optimistic).
- temporal — hold out the LAST 25% of maps in time. Validation is a genuinely
             later period; this is the stricter, honest generalization test.
The gap between the two numbers *is* the measure of temporal leakage.

Results live under output/real/<split>/ so they never touch the synthetic
smoke-test in output/.
"""
from pathlib import Path
import argparse
import copy

import numpy as np

from src.config import load_config
from src.data.load import load as load_nc
from src.train import train as train_unet
from src import evaluate as evaluate_mod
from src import robustness as robustness_mod

SPLITS = ["temporal", "random"]  # temporal first: it's the headline, honest test


def _split_cfg(cfg: dict, mode: str) -> dict:
    """Clone the config with output redirected to output/real/<mode>/."""
    c = copy.deepcopy(cfg)
    c["paths"]["output_dir"] = str(Path(cfg["paths"]["output_dir"]) / "real" / mode)
    return c


def _run_split(cfg: dict, mode: str, npy_path: Path,
               epochs: int, batch: int, lr: float, seed: int) -> dict:
    sc = _split_cfg(cfg, mode)
    Path(sc["paths"]["output_dir"]).mkdir(parents=True, exist_ok=True)
    print("\n" + "#" * 60)
    print(f"# HOLD-OUT = {mode.upper()}")
    print("#" * 60)
    print(f"train U-Net ({epochs} epochs)...")
    train_unet(sc, npy_path, epochs=epochs, batch=batch, lr=lr, seed=seed, split=mode)
    print("evaluate vs baseline...")
    summary = evaluate_mod.run(sc, npy_path, seed=seed, split=mode)
    print("robustness sweep...")
    rob = robustness_mod.run(sc, npy_path, seed=seed, split=mode)
    return {"summary": summary, "rob": rob}


def _robustness_table(rob: dict) -> str:
    covs = rob["coverages"]
    methods = ["mean", "nearest", "interpolation", "unet"]
    out = ("| hole coverage | mean | nearest | interpolation | **U-Net** |\n"
           "|---:|---:|---:|---:|---:|\n")
    for c in covs:
        cells = [f"{rob['rmse_mean'][m][c]:.3f}" for m in methods]
        out += f"| {c*100:.0f}% | {cells[0]} | {cells[1]} | {cells[2]} | **{cells[3]}** |\n"
    return out


def _write_results_md(cfg: dict, nc_path: Path, npy_path: Path,
                      maps: np.ndarray, results: dict, epochs: int) -> Path:
    out_root = Path(cfg["paths"]["output_dir"]) / "real"
    md = out_root / "RESULTS_REAL.md"
    region, t, var = cfg["region"], cfg["time"], cfg["variable"]
    n_nan = int(np.isnan(maps).sum())
    pct_nan = 100 * n_nan / maps.size if maps.size else 0.0

    def _headline_row(mode: str) -> str:
        s = results[mode]["summary"]
        return (f"| {mode} | {s['baseline_rmse_mean']:.3f} | {s['unet_rmse_mean']:.3f} "
                f"| {s['unet_mae_mean']:.3f} | {s['rmse_improvement_pct']:+.1f}% |")

    t_imp = results["temporal"]["summary"]["rmse_improvement_pct"]
    r_imp = results["random"]["summary"]["rmse_improvement_pct"]
    leak_note = (
        f"The random split reports **{r_imp:+.1f}%** and the temporal split "
        f"**{t_imp:+.1f}%**. The difference is the honest cost of temporal "
        f"autocorrelation: shuffled hourly frames are near-duplicates of training "
        f"frames, so the random number flatters the model. **Quote the temporal "
        f"number** as the real generalization result.")

    rob_sections = ""
    for mode in SPLITS:
        rob_sections += (f"### {mode} hold-out\n"
                         + _robustness_table(results[mode]["rob"]) + "\n")

    # Data-driven, honest interpretation — does the U-Net actually beat
    # interpolation under the STRICT temporal test, at any hole size?
    t_rob = results["temporal"]["rob"]
    covs = t_rob["coverages"]
    unet_beats_interp = any(
        t_rob["rmse_mean"]["unet"][c] < t_rob["rmse_mean"]["interpolation"][c]
        for c in covs)
    if t_imp >= 0 or unet_beats_interp:
        pattern_note = (
            "Under the temporal hold-out the U-Net still beats interpolation at "
            "larger gaps — the advantage survives the stricter test.")
        bottom_line = (
            f"**Bottom line:** the U-Net's edge holds out-of-time "
            f"({t_imp:+.1f}% headline), strongest where gaps are large.")
        margin_bullet = (
            "- **Modest margin:** on this smooth field interpolation is a strong "
            "baseline; the U-Net's edge is real but concentrated at larger gaps.")
    else:
        pattern_note = (
            "**Under the honest temporal hold-out, interpolation beats this U-Net "
            "at every gap size.** The U-Net's apparent win under the random split "
            "does not survive once the test set is genuinely later in time — it was "
            "an artifact of temporal leakage.")
        bottom_line = (
            f"**Bottom line:** with a single 10-day window this U-Net overfits the "
            f"training period and does **not** generalize to later sea states "
            f"({t_imp:+.1f}% out-of-time). The trustworthy deliverables here are the "
            f"honest measurement and the reproducible pipeline; beating "
            f"interpolation out-of-time needs more and more-diverse data "
            f"(multi-month, multi-region), not a code tweak.")
        margin_bullet = (
            "- **No out-of-time edge yet:** interpolation is a strong baseline on "
            "this smooth field; the U-Net only 'wins' under the leaky random split.")

    md.write_text(
        f"""# REAL-DATA Results — Marine Map Gap-Filling (Wave Height)

> Real data, not the synthetic smoke test. Evaluated under two hold-out schemes;
> the **temporal** one is the honest generalization result.

## Data provenance
- **Source file:** `{nc_path.name}`
- **Variable:** `{var['copernicus_var']}` ({var['short_name']}), significant wave height, meters.
- **Dataset id (config):** `{var.get('copernicus_dataset_id', 'n/a')}`
- **Region (config):** lon [{region['lon_min']}, {region['lon_max']}], lat [{region['lat_min']}, {region['lat_max']}] ({region['name']})
- **Time window (config):** {t['start']} -> {t['end']}
- **Processed array:** `{npy_path.name}`  shape `{tuple(int(x) for x in maps.shape)}` (maps, H, W)
- **Value range:** {float(np.nanmin(maps)):.2f} .. {float(np.nanmax(maps)):.2f} m
- **Missing (land/gaps):** {n_nan} pixels ({pct_nan:.1f}%)

## Headline — two hold-out schemes
| Hold-out | Baseline RMSE (m) | U-Net RMSE (m) | U-Net MAE (m) | Improvement |
|---|---:|---:|---:|---:|
{_headline_row("temporal")}
{_headline_row("random")}

**temporal** = train on the earliest ~75% of time steps, test on the latest ~25%
(genuinely later sea states). **random** = shuffle then split (optimistic).
Both score RMSE/MAE on hidden hole pixels only, land excluded, identical holes
for baseline and U-Net.

{leak_note}

## Robustness — RMSE vs hole size (vs a panel of baselines)
{rob_sections}For tiny gaps interpolation is essentially optimal in both schemes. {pattern_note}

{bottom_line}

## Honest limitations
- **Single 10-day window:** even the temporal hold-out tests only the last days
  of one download, not a different season. The right next step is a multi-month
  file so train and test cover different weather regimes.
{margin_bullet}
- Single region ({region['name']}), single variable ({var['copernicus_var']}),
  {maps.shape[1]}x{maps.shape[2]} resolution — intentional POC scope.

## Artifacts (under output/real/<split>/)
- `train_loss.png` — training/validation loss curve.
- `evaluation_example.png` — truth / holes / baseline / U-Net / per-pixel error.
- `robustness_rmse_vs_holesize.png` — RMSE vs hole size, all methods.
- `evaluation.json` , `robustness.json` — full numbers.

## How this was produced
```bash
python -m src.run_real {nc_path.as_posix()} --epochs {epochs}
```
"""
    )
    return md


def main(nc_path: str, epochs: int = 40, batch: int = 8, lr: float = 1e-3, seed: int = 0):
    cfg = load_config()

    print("=" * 60)
    print("STEP 1  load NetCDF -> NumPy grid")
    print("=" * 60)
    maps = load_nc(nc_path, cfg)
    size = cfg["preprocess"]["target_size"]
    npy_path = Path(cfg["paths"]["processed_dir"]) / (Path(nc_path).stem + f"_{size}x{size}.npy")
    if len(maps) < 16:
        raise SystemExit(
            f"Only {len(maps)} maps in this file — too few for a temporal hold-out. "
            "Download a longer time window in config.yaml."
        )

    results = {mode: _run_split(cfg, mode, npy_path, epochs, batch, lr, seed)
               for mode in SPLITS}

    md = _write_results_md(cfg, Path(nc_path), npy_path, maps, results, epochs)

    print("\n" + "=" * 60)
    print("SUMMARY (RMSE on hidden pixels, meters)")
    for mode in SPLITS:
        s = results[mode]["summary"]
        print(f"  {mode:9s}  baseline {s['baseline_rmse_mean']:.3f}  "
              f"U-Net {s['unet_rmse_mean']:.3f}  ({s['rmse_improvement_pct']:+.1f}%)")
    print(f"\nDONE. Real-data report: {md}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("nc_path", help="path to a real NetCDF file (e.g. data/raw/foo.nc)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args.nc_path, epochs=args.epochs, batch=args.batch, lr=args.lr, seed=args.seed)
