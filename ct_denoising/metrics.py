"""Standard image-quality metrics for normalized grayscale images."""

from __future__ import annotations

import math

import numpy as np
from skimage.metrics import structural_similarity


def mse(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Calculate mean squared error. Lower values are better."""
    _validate_pair(reference, estimate)

    reference_float = reference.astype(np.float64)
    estimate_float = estimate.astype(np.float64)

    return float(np.mean((reference_float - estimate_float) ** 2))


def psnr(
    reference: np.ndarray,
    estimate: np.ndarray,
    data_range: float = 1.0,
) -> float:
    """Calculate peak signal-to-noise ratio in decibels."""
    error = mse(reference, estimate)

    if error == 0.0:
        return float("inf")

    return float(10.0 * math.log10((data_range**2) / error))


def ssim(
    reference: np.ndarray,
    estimate: np.ndarray,
    data_range: float = 1.0,
) -> float:
    """Calculate standard windowed structural similarity."""
    _validate_pair(reference, estimate)

    reference_float = reference.astype(np.float64)
    estimate_float = estimate.astype(np.float64)

    return float(
        structural_similarity(
            reference_float,
            estimate_float,
            data_range=data_range,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
    )


def evaluate(
    reference: np.ndarray,
    estimate: np.ndarray,
) -> dict[str, float]:
    """Calculate all supported image-quality metrics."""
    return {
        "mse": mse(reference, estimate),
        "psnr_db": psnr(reference, estimate),
        "ssim": ssim(reference, estimate),
    }


def _validate_pair(
    reference: np.ndarray,
    estimate: np.ndarray,
) -> None:
    if reference.shape != estimate.shape:
        raise ValueError(
            f"Image shapes must match, got "
            f"{reference.shape} and {estimate.shape}"
        )

    if reference.size == 0:
        raise ValueError("Images cannot be empty")

    if reference.ndim != 2:
        raise ValueError(
            f"Expected 2D grayscale images, got shape {reference.shape}"
        )

    if not (
        np.isfinite(reference).all()
        and np.isfinite(estimate).all()
    ):
        raise ValueError("Images must contain only finite values")
