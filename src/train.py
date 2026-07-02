"""Week 3-4 — train the U-Net to fill holes in wave-height maps.

    python -m src.train --epochs 40

Loss is MSE over ocean pixels (land excluded). Saves the best model by
validation loss to output/unet.pt, the normalization stats to
output/norm_stats.json, and a loss curve to output/train_loss.png.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import load_config
from src.dataset import WaveInpaintingDataset, compute_stats
from src.model import UNet


def masked_mse(pred, target, mask):
    """MSE over pixels where mask == 1 (per-batch, averaged over valid pixels)."""
    diff2 = (pred - target) ** 2 * mask
    return diff2.sum() / mask.sum().clamp(min=1.0)


def split_maps(maps, val_frac=0.25, seed=0, mode="random"):
    """Hold out whole maps for validation so the net is scored on unseen fields.

    mode="random"   — shuffle then split (default; maximises train diversity).
    mode="temporal" — hold out the LAST maps in time. For an hourly time series
                      this is the honest, stricter test: validation is a genuinely
                      later period, not shuffled near-duplicates of training frames.
    """
    n_val = max(1, int(val_frac * len(maps)))
    if mode == "temporal":
        return maps[:-n_val], maps[-n_val:]
    idx = np.random.default_rng(seed).permutation(len(maps))
    return maps[idx[n_val:]], maps[idx[:n_val]]


def train(cfg, npy_path, epochs=40, batch=8, lr=1e-3, seed=0, split="random"):
    # Determinism: same seed -> same numbers, so reported metrics reproduce.
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)
    maps = np.load(npy_path)
    train_maps, val_maps = split_maps(maps, seed=seed, mode=split)
    mean, std = compute_stats(train_maps)  # stats from TRAIN only (no leakage)

    train_ds = WaveInpaintingDataset(train_maps, mean, std, augment=True)
    val_ds = WaveInpaintingDataset(val_maps, mean, std, augment=False)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = UNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  train={len(train_ds)} maps  val={len(val_ds)} maps")

    history, best_val = [], float("inf")
    for ep in range(1, epochs + 1):
        net.train()
        tr_loss = 0.0
        for b in train_dl:
            x, y, ocean = b["input"].to(device), b["target"].to(device), b["ocean"].to(device)
            opt.zero_grad()
            loss = masked_mse(net(x), y, ocean)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(x)
        tr_loss /= len(train_ds)

        net.eval()
        va_loss = 0.0
        with torch.no_grad():
            for b in val_dl:
                x, y, ocean = b["input"].to(device), b["target"].to(device), b["ocean"].to(device)
                va_loss += masked_mse(net(x), y, ocean).item() * len(x)
        va_loss /= len(val_ds)
        history.append((ep, tr_loss, va_loss))

        if va_loss < best_val:
            best_val = va_loss
            torch.save(net.state_dict(), out_dir / "unet.pt")
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"epoch {ep:3d}  train {tr_loss:.4f}  val {va_loss:.4f}"
                  f"{'  *' if va_loss == best_val else ''}")

    with open(out_dir / "norm_stats.json", "w") as f:
        json.dump({"mean": mean, "std": std, "npy_path": str(npy_path)}, f, indent=2)

    _plot_loss(history, out_dir / "train_loss.png")
    print(f"best val loss: {best_val:.4f}")
    print(f"saved: {out_dir/'unet.pt'} , {out_dir/'norm_stats.json'} , {out_dir/'train_loss.png'}")
    return best_val


def _plot_loss(history, path):
    ep, tr, va = zip(*history)
    plt.figure(figsize=(6, 4))
    plt.plot(ep, tr, label="train")
    plt.plot(ep, va, label="val")
    plt.xlabel("epoch"); plt.ylabel("masked MSE (normalized)"); plt.legend()
    plt.title("U-Net training")
    plt.savefig(path, dpi=100, bbox_inches="tight")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("npy_path", nargs="?", default=None)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    cfg = load_config()
    path = args.npy_path or sorted(Path(cfg["paths"]["processed_dir"]).glob("*_64x64.npy"))[0]
    train(cfg, path, epochs=args.epochs, batch=args.batch, lr=args.lr)
