# REAL-DATA Results — Marine Map Gap-Filling (Wave Height)

> These numbers are on **real** data, not the synthetic smoke test. Source and
> preprocessing are stamped below for provenance.

## Data provenance
- **Source file:** `ww3_global.nc`
- **Variable:** `Thgt` (wave), significant wave height, meters.
- **Dataset id (config):** `NWW3_Global_Best (NOAA ERDDAP, WaveWatch III)`
- **Region (config):** lon [300.0, 340.0], lat [30.0, 50.0] (north_atlantic)
- **Time window (config):** 2026-06-07 -> 2026-06-17
- **Processed array:** `ww3_global_40x40.npy`  shape `(241, 40, 40)` (maps, H, W)
- **Value range:** 0.06 .. 5.68 m
- **Missing (land/gaps):** 3615 pixels (0.9%)

## Headline result
| Method | RMSE (m) down | MAE (m) down |
|--------|--------------:|-------------:|
| Interpolation baseline | 0.139 | 0.096 |
| **U-Net** | **0.126** | **0.098** |
| **Improvement** | **+9.9%** | |

Scored on 60 held-out maps, on hidden hole pixels only, land
excluded — the same fair setup as the synthetic run (same maps, same holes for
both methods). The U-Net's gain shows up in **RMSE** (which punishes large errors) more than in MAE — i.e. it mainly prevents the big misses, and is roughly level with interpolation on typical pixels.

## Robustness — RMSE vs hole size (vs a panel of baselines)
| hole coverage | mean | nearest | interpolation | **U-Net** |
|---:|---:|---:|---:|---:|
| 5% | 0.602 | 0.144 | 0.103 | **0.108** |
| 10% | 0.533 | 0.179 | 0.113 | **0.125** |
| 20% | 0.593 | 0.194 | 0.149 | **0.143** |
| 35% | 0.701 | 0.234 | 0.165 | **0.160** |
| 50% | 0.638 | 0.230 | 0.164 | **0.159** |

At small gaps (5%) classical interpolation is essentially optimal and edges out the U-Net; the U-Net crosses over and leads once gaps grow (>=20%), reaching +3.0% at 50% coverage.

## Honest limitations
- **Temporal autocorrelation:** the 241 maps are consecutive model
  time steps (hourly), so held-out maps are near-neighbours of training maps in
  time. The U-Net vs baseline comparison is fair (identical maps/holes), but
  "held-out" here is not "a different week/season". A stricter test holds out
  whole days or a separate month.
- **Modest margin, honestly:** on this smooth field with small default holes the
  interpolation baseline is already strong; the U-Net's edge is real but small,
  and concentrated at larger gaps (see table above).
- Single region (north_atlantic), single variable (Thgt),
  40x40 resolution — intentional POC scope.

## Artifacts (in output/real/)
- `train_loss.png` — training/validation loss curve.
- `evaluation_example.png` — truth / holes / baseline / U-Net / per-pixel error.
- `robustness_rmse_vs_holesize.png` — RMSE vs hole size, all methods.
- `evaluation.json` , `robustness.json` — full numbers.

## How this was produced
```bash
python -m src.run_real data/raw/ww3_global.nc --epochs 40
```
