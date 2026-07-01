"""Week 2 — create artificial holes ("clouds") over complete maps.

This turns one complete map into a supervised training pair:
  - input  = map with a hole (values inside the hole removed)
  - target = the original complete map (the correct answer)

A hole is a boolean mask (True = pixel to hide). We only ever hide pixels
that had real data, so land/existing-NaN never counts as a "hole to score".

    python -m src.masks   # quick self-test + preview
"""
import numpy as np


def random_rectangles(shape, rng, n=(1, 3), frac=(0.1, 0.25)):
    """Mask made of a few random rectangles.

    shape : (H, W) of the map.
    n     : (min, max) number of rectangles.
    frac  : (min, max) side length of each rectangle as a fraction of the map.
    Returns a boolean array, True where a hole is punched.
    """
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    for _ in range(rng.integers(n[0], n[1] + 1)):
        rh = int(h * rng.uniform(*frac))
        rw = int(w * rng.uniform(*frac))
        top = rng.integers(0, max(1, h - rh))
        left = rng.integers(0, max(1, w - rw))
        mask[top:top + rh, left:left + rw] = True
    return mask


def random_blob(shape, rng, coverage=0.2):
    """An irregular blob mask grown from a seed until ~coverage is reached.

    Produces more cloud-like holes than plain rectangles.
    """
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    target = int(coverage * h * w)
    y, x = rng.integers(0, h), rng.integers(0, w)
    frontier = [(y, x)]
    mask[y, x] = True
    count = 1
    while frontier and count < target:
        cy, cx = frontier.pop(rng.integers(0, len(frontier)))
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and not mask[ny, nx]:
                if rng.random() < 0.6:  # stochastic growth -> ragged edges
                    mask[ny, nx] = True
                    frontier.append((ny, nx))
                    count += 1
    return mask


def apply_mask(complete_map, hole_mask):
    """Return (corrupted_map, eval_mask).

    corrupted_map : copy of the map with holes set to NaN (model input).
    eval_mask     : True only where a hole overlaps originally-valid data,
                    i.e. the pixels we can actually score against truth.
    """
    valid = ~np.isnan(complete_map)
    eval_mask = hole_mask & valid
    corrupted = complete_map.copy()
    corrupted[hole_mask] = np.nan
    return corrupted, eval_mask


def make_hole(shape, rng, kind="mixed", **kw):
    """Dispatch helper: 'rect', 'blob', or 'mixed' (random choice)."""
    if kind == "rect":
        return random_rectangles(shape, rng, **kw)
    if kind == "blob":
        return random_blob(shape, rng, **kw)
    if kind == "mixed":
        return (random_rectangles(shape, rng) if rng.random() < 0.5
                else random_blob(shape, rng))
    raise ValueError(f"unknown kind: {kind}")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    m = make_hole((64, 64), rng, kind="blob")
    print(f"blob mask covers {100 * m.mean():.1f}% of the map")
    fake = np.ones((64, 64), dtype=np.float32)
    corrupted, eval_mask = apply_mask(fake, m)
    print(f"corrupted NaNs: {np.isnan(corrupted).sum()}, "
          f"scorable pixels: {eval_mask.sum()}")
