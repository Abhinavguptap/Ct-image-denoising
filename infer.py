"""Denoise one grayscale image with a trained residual DnCNN model."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from ct_denoising.data import read_grayscale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model, compile=False)
    noisy = read_grayscale(args.input)
    predicted_noise = model.predict(noisy[None, ..., None], verbose=0)[0, ..., 0]
    denoised = np.clip(noisy - predicted_noise, 0.0, 1.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), np.round(denoised * 255).astype(np.uint8)):
        raise OSError(f"Could not write output image: {args.output}")
    print(f"Saved denoised image to {args.output}")


if __name__ == "__main__":
    main()
