import numpy as np
import pytest

from ct_denoising.metrics import mse, psnr, ssim


def test_identical_images_have_perfect_metrics():
    image = np.full((16, 16), 0.5, dtype=np.float32)

    assert mse(image, image) == 0.0
    assert psnr(image, image) == float("inf")
    assert ssim(image, image) == pytest.approx(1.0)


def test_metrics_detect_degradation():
    reference = np.zeros((16, 16), dtype=np.float32)
    estimate = np.full((16, 16), 0.1, dtype=np.float32)

    assert mse(reference, estimate) == pytest.approx(0.01)
    assert psnr(reference, estimate) == pytest.approx(20.0)
    assert ssim(reference, estimate) < 1.0


def test_shape_mismatch_is_rejected():
    reference = np.zeros((16, 16), dtype=np.float32)
    estimate = np.zeros((20, 20), dtype=np.float32)

    with pytest.raises(ValueError, match="shapes must match"):
        mse(reference, estimate)
