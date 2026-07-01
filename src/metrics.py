"""Shared evaluation metrics — scored ONLY on the hidden hole pixels."""
import numpy as np


def rmse(pred, truth, mask):
    """Root mean squared error over pixels where mask is True."""
    d = (pred[mask] - truth[mask])
    return float(np.sqrt(np.mean(d * d)))


def mae(pred, truth, mask):
    """Mean absolute error over pixels where mask is True."""
    return float(np.mean(np.abs(pred[mask] - truth[mask])))


def evaluate(pred, truth, mask):
    """Return both metrics plus the count of scored pixels."""
    return {"rmse": rmse(pred, truth, mask),
            "mae": mae(pred, truth, mask),
            "n_pixels": int(mask.sum())}
