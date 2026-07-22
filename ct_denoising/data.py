"""Data validation, source-level splitting, and patch generation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def list_images(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    paths = sorted(path for path in root.iterdir() if path.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not paths:
        raise ValueError(f"No supported images found in {root}")
    return paths


def split_sources(paths: Iterable[Path], seed: int = 42, train_fraction: float = 0.67) -> tuple[list[Path], list[Path], list[Path]]:
    items = list(paths)
    if len(items) < 3:
        raise ValueError("At least three source images are required for train/validation/test splits")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    order = np.random.default_rng(seed).permutation(len(items))
    shuffled = [items[index] for index in order]
    train_count = max(1, min(len(items) - 2, int(len(items) * train_fraction)))
    remaining = len(items) - train_count
    validation_count = max(1, remaining // 2)
    return shuffled[:train_count], shuffled[train_count : train_count + validation_count], shuffled[train_count + validation_count :]


def read_grayscale(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image.astype(np.float32) / 255.0


def add_gaussian_noise(clean: np.ndarray, sigma: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    noise = rng.normal(0.0, sigma / 255.0, size=clean.shape).astype(np.float32)
    noisy = np.clip(clean + noise, 0.0, 1.0)
    residual = noisy - clean
    return noisy, residual


def make_patches(paths: Iterable[Path], patch_size: int, patches_per_image: int, sigma: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if patch_size <= 0 or patches_per_image <= 0:
        raise ValueError("Patch configuration must be positive")
    rng = np.random.default_rng(seed)
    noisy_patches, residual_patches = [], []
    for path in paths:
        image = read_grayscale(path)
        height, width = image.shape
        if min(height, width) < patch_size:
            scale = patch_size / min(height, width)
            image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_CUBIC)
            height, width = image.shape
        for _ in range(patches_per_image):
            top = int(rng.integers(0, height - patch_size + 1))
            left = int(rng.integers(0, width - patch_size + 1))
            clean = image[top : top + patch_size, left : left + patch_size]
            if rng.random() < 0.5:
                clean = np.fliplr(clean)
            rotation = int(rng.integers(0, 4))
            clean = np.rot90(clean, rotation).copy()
            noisy, residual = add_gaussian_noise(clean, sigma, rng)
            noisy_patches.append(noisy[..., None])
            residual_patches.append(residual[..., None])
    return np.asarray(noisy_patches, dtype=np.float32), np.asarray(residual_patches, dtype=np.float32)
