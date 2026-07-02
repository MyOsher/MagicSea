# REAL-DATA Results — Marine Map Gap-Filling (Wave Height)

> Real data, not the synthetic smoke test. Evaluated under two hold-out schemes;
> the **temporal** one is the honest generalization result.

## Data provenance
- **Source file:** `ww3_6month_daily.nc`
- **Variable:** `Thgt` (wave), significant wave height, meters.
- **Dataset id (config):** `NWW3_Global_Best (NOAA ERDDAP, WaveWatch III)`
- **Region (config):** lon [300.0, 340.0], lat [30.0, 50.0] (north_atlantic)
- **Time window (config):** 2026-01-07 -> 2026-06-17
- **Processed array:** `ww3_6month_daily_40x40.npy`  shape `(162, 40, 40)` (maps, H, W)
- **Value range:** 0.00 .. 15.18 m
- **Missing (land/gaps):** 2431 pixels (0.9%)

## Headline — two hold-out schemes
| Hold-out | Baseline RMSE (m) | U-Net RMSE (m) | U-Net MAE (m) | Improvement |
|---|---:|---:|---:|---:|
| temporal | 0.158 | 0.191 | 0.144 | -20.3% |
| random | 0.184 | 0.201 | 0.152 | -9.2% |

**temporal** = train on the earliest ~75% of time steps, test on the latest ~25%
(genuinely later sea states). **random** = shuffle then split (optimistic).
Both score RMSE/MAE on hidden hole pixels only, land excluded, identical holes
for baseline and U-Net.

The random split reports **-9.2%** and the temporal split **-20.3%**. Shuffling lets near-in-time frames leak between train and test, so the random number is the optimistic one. **Quote the temporal number** as the real generalization result.

## Robustness — RMSE vs hole size (vs a panel of baselines)
### temporal hold-out
| hole coverage | mean | nearest | interpolation | **U-Net** |
|---:|---:|---:|---:|---:|
| 5% | 0.834 | 0.201 | 0.116 | **0.158** |
| 10% | 0.778 | 0.234 | 0.139 | **0.200** |
| 20% | 0.813 | 0.248 | 0.167 | **0.214** |
| 35% | 0.945 | 0.288 | 0.203 | **0.241** |
| 50% | 0.998 | 0.318 | 0.215 | **0.267** |

### random hold-out
| hole coverage | mean | nearest | interpolation | **U-Net** |
|---:|---:|---:|---:|---:|
| 5% | 0.997 | 0.226 | 0.141 | **0.209** |
| 10% | 1.104 | 0.270 | 0.150 | **0.214** |
| 20% | 1.032 | 0.285 | 0.177 | **0.215** |
| 35% | 1.176 | 0.331 | 0.214 | **0.253** |
| 50% | 1.305 | 0.408 | 0.286 | **0.296** |

For tiny gaps interpolation is essentially optimal in both schemes. **Under the honest temporal hold-out, interpolation beats this U-Net at every gap size.** On a genuinely later test period the U-Net does not out-predict classical interpolation on this smooth field.

**Bottom line:** on this ~161-day window, 162 maps (~122 for training) the U-Net does **not** beat classical interpolation out-of-time (-20.3%). Even with a real seasonal hold-out, a compact model trained on a modest number of maps can't out-predict interpolation here. The value delivered is the honest measurement and the reproducible pipeline; closing the gap needs more data (more regions/years) and/or a stronger model — not a quick tweak.

## Honest limitations
- **Scale — ~161-day window, 162 maps (~122 for training):** one region, one download. A compact U-Net on this many
  maps is data-limited; more regions and years would give it patterns that
  generalize beyond a single area/period.
- **No edge yet:** interpolation beats the U-Net under **both** schemes (-9.2% random, -20.3% temporal) — the random split is just the less-pessimistic of the two, not a win.
- **Training-length sensitive:** shown at 200 epochs; on this modest set
  fewer epochs underfit badly and widen the gap to interpolation. More data would
  reduce this sensitivity.
- **Temporal split is optimistic:** its held-out period also drives best-checkpoint
  selection in training, so the temporal number flatters the U-Net — and it still
  does not beat interpolation.
- Single region (north_atlantic), single variable (Thgt),
  40x40 resolution — intentional POC scope.

## Artifacts (under output/real/<split>/)
- `train_loss.png` — training/validation loss curve.
- `evaluation_example.png` — truth / holes / baseline / U-Net / per-pixel error.
- `robustness_rmse_vs_holesize.png` — RMSE vs hole size, all methods.
- `evaluation.json` , `robustness.json` — full numbers.

## How this was produced
```bash
python -m src.run_real data/raw/ww3_6month_daily.nc --epochs 200
```
