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

## Robustness — RMSE vs hole size (vs a panel of baselines)
The advantage isn't a single number: the U-Net's lead over interpolation **grows
as gaps get larger**, exactly where naive methods fail. RMSE (m) on hidden pixels:

| hole coverage | mean | nearest | interpolation | **U-Net** |
|--------------:|-----:|--------:|--------------:|----------:|
| 5%  | 0.90 | 0.24 | 0.117 | **0.112** |
| 10% | 0.93 | 0.28 | 0.143 | **0.112** |
| 20% | 0.88 | 0.28 | 0.147 | **0.138** |
| 35% | 0.95 | 0.35 | 0.195 | **0.167** |
| 50% | 0.91 | 0.34 | 0.204 | **0.164** |

At 50% coverage the U-Net beats interpolation by ~20% RMSE (vs ~4% at 5%).
See `robustness_rmse_vs_holesize.png`.

## Artifacts
- `train_loss.png` — training/validation loss curve.
- `evaluation_example.png` — truth / holes / baseline / U-Net / per-pixel error.
- `robustness_rmse_vs_holesize.png` — RMSE vs hole size, all methods.
- `evaluation.json` , `robustness.json` — full numbers.
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
