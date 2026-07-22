# Dataset card

## Contents

This repository contains nine images in `Clean/` and nine images in `noisy/`. The files are a small demonstration set and are not sufficient for clinical or population-level conclusions.

## Alignment audit

Files with the same name do not consistently share the same pixel dimensions. Several pairs also appear to have different crops. They must therefore not be treated as registered pixel-to-pixel ground truth pairs without a documented registration procedure and visual quality control.

The training pipeline uses only `Clean/` as source images and creates seeded synthetic Gaussian noise. The original `noisy/` directory is reserved for qualitative inference.

## Provenance and licensing

The original repository does not document the source, collection process, patient demographics, scanner protocol, de-identification procedure, or license of these images. Before redistributing or using the data beyond this demonstration, the repository owner should verify that they have permission and add authoritative provenance/licensing information.

## Known limitations

- Nine source images provide very low diversity.
- PNG/JPEG files discard CT acquisition metadata and calibrated Hounsfield units.
- Synthetic additive Gaussian noise is not a complete model of low-dose CT noise.
- Patient-level splitting cannot be verified because patient identifiers are unavailable.
- Image annotations, compression, screenshots, or preprocessing may introduce shortcuts.

## Recommended next dataset

For credible research, replace this sample with a licensed, de-identified low-dose CT benchmark containing aligned normal-dose/low-dose data. Split by patient—not slice—and document acquisition protocols, preprocessing, registration, exclusions, and ethics/licensing constraints.
## Source and provenance

The original source and redistribution licence of the demonstration images could not be verified. The repository owner did not collect the images directly and does not claim ownership of them.

The images are retained temporarily to document the original educational prototype. They must not be reused, redistributed or treated as a clinical dataset without independently verifying their source, licence, de-identification status and permitted use.

A future version of this project should replace these files with samples from a clearly licensed and de-identified public CT benchmark.
