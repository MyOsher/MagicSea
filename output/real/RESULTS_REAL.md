# REAL-DATA Results — Marine Map Gap-Filling (Wave Height)

> Real data, not the synthetic smoke test. Evaluated under two hold-out schemes;
> the **temporal** one is the honest generalization result.

## Data provenance
- **Source file:** `ww3_global.nc`
- **Variable:** `Thgt` (wave), significant wave height, meters.
- **Dataset id (config):** `NWW3_Global_Best (NOAA ERDDAP, WaveWatch III)`
- **Region (config):** lon [300.0, 340.0], lat [30.0, 50.0] (north_atlantic)
- **Time window (config):** 2026-06-07 -> 2026-06-17
- **Processed array:** `ww3_global_40x40.npy`  shape `(241, 40, 40)` (maps, H, W)
- **Value range:** 0.06 .. 5.68 m
- **Missing (land/gaps):** 3615 pixels (0.9%)

## Headline — two hold-out schemes
| Hold-out | Baseline RMSE (m) | U-Net RMSE (m) | U-Net MAE (m) | Improvement |
|---|---:|---:|---:|---:|
| temporal | 0.103 | 0.120 | 0.091 | -16.3% |
| random | 0.139 | 0.132 | 0.100 | +5.3% |

**temporal** = train on the earliest ~75% of time steps, test on the latest ~25%
(genuinely later sea states). **random** = shuffle then split (optimistic).
Both score RMSE/MAE on hidden hole pixels only, land excluded, identical holes
for baseline and U-Net.

The random split reports **+5.3%** and the temporal split **-16.3%**. The difference is the honest cost of temporal autocorrelation: shuffled hourly frames are near-duplicates of training frames, so the random number flatters the model. **Quote the temporal number** as the real generalization result.

## Robustness — RMSE vs hole size (vs a panel of baselines)
### temporal hold-out
| hole coverage | mean | nearest | interpolation | **U-Net** |
|---:|---:|---:|---:|---:|
| 5% | 0.413 | 0.112 | 0.075 | **0.114** |
| 10% | 0.404 | 0.140 | 0.094 | **0.121** |
| 20% | 0.434 | 0.151 | 0.107 | **0.135** |
| 35% | 0.473 | 0.170 | 0.125 | **0.151** |
| 50% | 0.496 | 0.176 | 0.132 | **0.159** |

### random hold-out
| hole coverage | mean | nearest | interpolation | **U-Net** |
|---:|---:|---:|---:|---:|
| 5% | 0.602 | 0.144 | 0.103 | **0.111** |
| 10% | 0.533 | 0.179 | 0.113 | **0.122** |
| 20% | 0.593 | 0.194 | 0.149 | **0.142** |
| 35% | 0.701 | 0.234 | 0.165 | **0.163** |
| 50% | 0.638 | 0.230 | 0.164 | **0.160** |

For tiny gaps interpolation is essentially optimal in both schemes. **Under the honest temporal hold-out, interpolation beats this U-Net at every gap size.** The U-Net's apparent win under the random split does not survive once the test set is genuinely later in time — it was an artifact of temporal leakage.

**Bottom line:** with a single 10-day window this U-Net overfits the training period and does **not** generalize to later sea states (-16.3% out-of-time). The trustworthy deliverables here are the honest measurement and the reproducible pipeline; beating interpolation out-of-time needs more and more-diverse data (multi-month, multi-region), not a code tweak.

## Honest limitations
- **Single 10-day window:** even the temporal hold-out tests only the last days
  of one download, not a different season. The right next step is a multi-month
  file so train and test cover different weather regimes.
- **No out-of-time edge yet:** interpolation is a strong baseline on this smooth field; the U-Net only 'wins' under the leaky random split.
- Single region (north_atlantic), single variable (Thgt),
  40x40 resolution — intentional POC scope.

## Artifacts (under output/real/<split>/)
- `train_loss.png` — training/validation loss curve.
- `evaluation_example.png` — truth / holes / baseline / U-Net / per-pixel error.
- `robustness_rmse_vs_holesize.png` — RMSE vs hole size, all methods.
- `evaluation.json` , `robustness.json` — full numbers.

## How this was produced
```bash
python -m src.run_real data/raw/ww3_global.nc --epochs 40
```
