"""Framework-independent image quality metrics for normalized grayscale images."""

from __future__ import annotations

import math

import numpy as np


def mse(reference: np.ndarray, estimate: np.ndarray) -> float:
    _validate_pair(reference, estimate)
    return float(np.mean((reference.astype(np.float64) - estimate.astype(np.float64)) ** 2))


def psnr(reference: np.ndarray, estimate: np.ndarray, data_range: float = 1.0) -> float:
    error = mse(reference, estimate)
    if error == 0.0:
        return float("inf")
    return float(10.0 * math.log10((data_range**2) / error))


def ssim(reference: np.ndarray, estimate: np.ndarray, data_range: float = 1.0) -> float:
    """Compute global SSIM without an additional runtime dependency.

    Global SSIM is deterministic and adequate for experiment comparison. A
    production imaging study should additionally report windowed/multiscale SSIM.
    """
    _validate_pair(reference, estimate)
    x = reference.astype(np.float64)
    y = estimate.astype(np.float64)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mean_x, mean_y = float(x.mean()), float(y.mean())
    var_x, var_y = float(x.var()), float(y.var())
    covariance = float(np.mean((x - mean_x) * (y - mean_y)))
    numerator = (2 * mean_x * mean_y + c1) * (2 * covariance + c2)
    denominator = (mean_x**2 + mean_y**2 + c1) * (var_x + var_y + c2)
    return float(numerator / denominator)


def evaluate(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    return {"mse": mse(reference, estimate), "psnr_db": psnr(reference, estimate), "ssim": ssim(reference, estimate)}


def _validate_pair(reference: np.ndarray, estimate: np.ndarray) -> None:
    if reference.shape != estimate.shape:
        raise ValueError(f"Image shapes must match, got {reference.shape} and {estimate.shape}")
    if reference.size == 0:
        raise ValueError("Images cannot be empty")
    if not (np.isfinite(reference).all() and np.isfinite(estimate).all()):
        raise ValueError("Images must contain only finite values")
