"""Train residual DnCNN and evaluate against a noisy-input baseline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import tensorflow as tf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ct_denoising.data import add_gaussian_noise, list_images, make_patches, read_grayscale, split_sources
from ct_denoising.metrics import evaluate
from ct_denoising.model import build_dncnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dir", type=Path, default=Path("Clean"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--patches-per-image", type=int, default=128)
    parser.add_argument("--noise-sigma", type=float, default=25.0, help="Gaussian noise std. dev. on the 0-255 scale")
    parser.add_argument("--depth", type=int, default=17)
    parser.add_argument("--filters", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    set_determinism(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sources = list_images(args.clean_dir)
    train_sources, validation_sources, test_sources = split_sources(sources, seed=args.seed)
    config = vars(args).copy()
    config.update(
        train_sources=[path.name for path in train_sources],
        validation_sources=[path.name for path in validation_sources],
        test_sources=[path.name for path in test_sources],
    )
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    train_x, train_y = make_patches(train_sources, args.patch_size, args.patches_per_image, args.noise_sigma, args.seed)
    validation_x, validation_y = make_patches(validation_sources, args.patch_size, args.patches_per_image, args.noise_sigma, args.seed + 1)

    model = build_dncnn(args.depth, args.filters)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate), loss="mse")
    checkpoint = args.output_dir / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
        tf.keras.callbacks.CSVLogger(args.output_dir / "training_history.csv"),
    ]
    history = model.fit(train_x, train_y, validation_data=(validation_x, validation_y), batch_size=args.batch_size, epochs=args.epochs, callbacks=callbacks, verbose=2)

    plot_history(history.history, args.output_dir / "training_curves.png")
    results = evaluate_model(model, test_sources, args.noise_sigma, args.seed + 2, args.output_dir / "comparisons")
    (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["aggregate"], indent=2))


def evaluate_model(model: tf.keras.Model, test_sources: list[Path], sigma: float, seed: int, comparison_dir: Path) -> dict:
    comparison_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    records = []
    for path in test_sources:
        clean = read_grayscale(path)
        noisy, _ = add_gaussian_noise(clean, sigma, rng)
        predicted_noise = model.predict(noisy[None, ..., None], verbose=0)[0, ..., 0]
        denoised = np.clip(noisy - predicted_noise, 0.0, 1.0)
        baseline = evaluate(clean, noisy)
        trained = evaluate(clean, denoised)
        records.append({"image": path.name, "noisy_baseline": baseline, "dncnn": trained})
        save_comparison(clean, noisy, denoised, comparison_dir / f"{path.stem}_comparison.png")
    aggregate = {}
    for method in ("noisy_baseline", "dncnn"):
        aggregate[method] = {metric: float(np.mean([row[method][metric] for row in records])) for metric in ("mse", "psnr_db", "ssim")}
    aggregate["improvement"] = {
        "psnr_db": aggregate["dncnn"]["psnr_db"] - aggregate["noisy_baseline"]["psnr_db"],
        "ssim": aggregate["dncnn"]["ssim"] - aggregate["noisy_baseline"]["ssim"],
        "mse_reduction_percent": 100.0 * (1.0 - aggregate["dncnn"]["mse"] / aggregate["noisy_baseline"]["mse"]),
    }
    return {"evaluation_protocol": f"Held-out source images with seeded Gaussian noise sigma={sigma}/255", "per_image": records, "aggregate": aggregate}


def plot_history(history: dict[str, list[float]], output: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(history["loss"], label="train")
    plt.plot(history["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("DnCNN training curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def save_comparison(clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, image, title in zip(axes, (clean, noisy, denoised), ("Clean", "Synthetic noisy", "DnCNN denoised")):
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
