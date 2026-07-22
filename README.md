# Reproducible CT Image Denoising with DnCNN

An end-to-end TensorFlow implementation of **residual DnCNN** for grayscale CT image denoising. The project emphasizes reproducible experiments, leakage-resistant evaluation, quantitative baselines, and honest reporting on a very small demonstration dataset.

> **Research-use disclaimer:** This is an educational prototype, not a medical device. It has not been clinically validated and must not be used for diagnosis or patient care.

## Why this version is methodologically safer

The repository contains nine clean images and nine separately collected noisy images. Corresponding files have different dimensions/crops, so they are **not pixel-aligned pairs**. Training directly on them would make pixel-wise loss and PSNR/SSIM evaluation invalid.

This pipeline therefore:

- splits data **by source image before extracting patches**, preventing near-duplicate patch leakage;
- adds seeded Gaussian noise to clean images for supervised residual learning;
- predicts noise and reconstructs the clean image as `noisy - predicted_noise`;
- compares the trained model with the unprocessed noisy-input baseline;
- reports MSE, PSNR, and SSIM on a held-out source image;
- saves configuration, metrics, checkpoints, curves, and visual comparisons.

The original `noisy/` images remain available for qualitative inference only. Claims about real low-dose CT performance require a larger, properly registered dataset.

## Architecture

The model follows the residual-learning idea from DnCNN:

1. Conv + ReLU input block
2. Configurable Conv + BatchNorm + ReLU blocks
3. Final Conv layer predicting the noise residual

Defaults use 17 convolutional layers, 64 feature maps, and 3×3 kernels. Configuration is exposed through command-line arguments.

## Repository structure

```text
.
├── Clean/                  # nine clean demonstration images
├── noisy/                  # unregistered real noisy images (inference only)
├── ct_denoising/
│   ├── data.py             # deterministic splits, patches, synthetic noise
│   ├── metrics.py          # MSE, PSNR, SSIM
│   └── model.py            # residual DnCNN
├── tests/                  # fast unit tests
├── train.py                # train + held-out evaluation
├── infer.py                # denoise one image
├── requirements.txt
└── .github/workflows/ci.yml
```

## Setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Train and evaluate

```bash
python train.py --clean-dir Clean --output-dir artifacts --epochs 50
```

Useful options:

```bash
python train.py --help
python train.py --noise-sigma 25 --patch-size 64 --patches-per-image 128 --seed 42
```

Artifacts include:

- `best_model.keras` — best validation checkpoint;
- `config.json` — exact experiment configuration;
- `metrics.json` — per-image and aggregate baseline/model metrics;
- `training_history.csv` and `training_curves.png`;
- `comparisons/` — clean, noisy, and denoised held-out examples.

Metrics are intentionally not hard-coded into this README. Run the experiment on your machine and report the generated values. With only nine images, results demonstrate engineering methodology—not clinical generalization.

## Inference

```bash
python infer.py --model artifacts/best_model.keras --input noisy/img1.png --output prediction.png
```

## Tests

```bash
pytest -q
```

CI performs syntax checks and tests the framework-independent metric and split logic. Full TensorFlow training is kept outside CI to avoid expensive, non-deterministic hosted runs.

## Responsible interpretation

- Dataset size is extremely small.
- The supplied noisy images are not registered to the clean images.
- Synthetic Gaussian noise does not fully represent low-dose CT acquisition noise.
- Evaluation is image-level and leakage-resistant, but statistical confidence remains limited.
- A production study should use DICOM-aware preprocessing, scanner/protocol metadata, a registered public low-dose CT dataset, multiple random seeds, confidence intervals, and radiologist review.

## Resume-safe description

> Built a reproducible residual DnCNN pipeline for grayscale CT denoising with source-level data splits, seeded synthetic-noise generation, automated PSNR/SSIM/MSE evaluation, baseline comparisons, checkpointing, tests, and CI; documented limitations of a nine-image prototype dataset.

Add measured improvements only after running the experiment, preserving `metrics.json`, and verifying the values.

## Reference

Zhang et al., “Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising,” IEEE Transactions on Image Processing, 2017.
