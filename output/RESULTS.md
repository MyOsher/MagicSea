# POC Results — Marine Map Gap-Filling (Wave Height)

> **Note:** numbers below are on **synthetic** wave-height data (a smoke test that
> proves the pipeline end-to-end). Re-run on real Copernicus data before quoting
> these to customers or investors.

## Setup
- **Variable:** significant wave height (VHM0), meters.
- **Task:** fill artificial holes (rectangular + cloud-like blobs) in 64×64 maps.
- **Data:** 200 synthetic daily maps → 150 train / 50 held-out (whole maps held out).
- **Metric:** RMSE / MAE on the hidden hole pixels only. Land excluded.

## Headline result
| Method | RMSE (m) ↓ | MAE (m) ↓ |
|--------|-----------|-----------|
| Interpolation baseline | 0.178 | — |
| **U-Net** | **0.126** | — |
| **Improvement** | **+29.2%** | |

The U-Net reduces reconstruction error by ~29% over classical interpolation on
maps it never saw during training. Training and validation loss track each other
closely (no overfitting).

## Artifacts
- `train_loss.png` — training/validation loss curve.
- `evaluation_example.png` — truth / holes / baseline / U-Net / per-pixel error.
- `evaluation.json` — full per-map numbers for both methods.
- `baseline_metrics.json` — baseline-only reference.

## How to reproduce
```bash
pip install -r requirements.txt
python -m src.data.synthetic --days 200
python -m src.data.load data/raw/mediterranean_wave_SYNTHETIC.nc
python -m src.train --epochs 40
python -m src.evaluate
```

## Honest limitations
- Synthetic data is smooth and low-diversity; real wave fields have sharper
  fronts and storms — expect the gap to both methods' errors to grow.
- Single region, single variable, 64×64 resolution (intentional POC scope).
- Next validation step is the real test: same pipeline on Copernicus VHM0.
