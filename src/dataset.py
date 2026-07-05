"""Week 3-4 — turn processed maps into (input, target) training pairs.

Each sample:
  input  : [filled_normalized_map, valid_mask]   (2, H, W)
  target : normalized complete map                (1, H, W)
  ocean  : where truth exists (loss is scored here, land excluded)
  hole   : the artificially hidden pixels         (for held-out scoring)

Holes are sampled fresh each access when augment=True (data augmentation).
When augment=False the hole is deterministic per index, so validation and
the baseline can be compared on identical holes.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from src.masks import make_hole, apply_mask


def compute_stats(maps):
    """Mean/std over all ocean (non-NaN) pixels of a stack of maps."""
    vals = maps[~np.isnan(maps)]
    return float(vals.mean()), float(vals.std() + 1e-6)


class WaveInpaintingDataset(Dataset):
    def __init__(self, maps, mean, std, kind="mixed", augment=True, base_seed=0):
        self.maps = maps.astype(np.float32)
        self.mean, self.std = mean, std
        self.kind, self.augment, self.base_seed = kind, augment, base_seed
        # Persistent seeded generator for training augmentation: holes still vary
        # across samples/epochs, but the whole run reproduces given a fixed seed
        # (the DataLoader shuffle order is itself seeded via torch.manual_seed).
        self.aug_rng = np.random.default_rng(base_seed + 104729)

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, idx):
        truth = self.maps[idx]
        ocean = ~np.isnan(truth)

        # Deterministic holes for val/eval; seeded-but-varied for training.
        rng = self.aug_rng if self.augment else np.random.default_rng(self.base_seed + idx)
        hole = make_hole(truth.shape, rng, kind=self.kind) & ocean
        corrupted, _ = apply_mask(truth, hole)

        norm = (truth - self.mean) / self.std
        target = np.where(ocean, norm, 0.0).astype(np.float32)

        # Model input: known pixels only (holes AND land zeroed out), plus a
        # mask channel telling the net which pixels are trustworthy.
        known = ocean & ~hole
        filled = np.where(known, norm, 0.0).astype(np.float32)
        inp = np.stack([filled, known.astype(np.float32)], axis=0)

        return {
            "input": torch.from_numpy(inp),
            "target": torch.from_numpy(target)[None],
            "ocean": torch.from_numpy(ocean.astype(np.float32))[None],
            "hole": torch.from_numpy(hole.astype(np.float32))[None],
        }
