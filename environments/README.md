# Environment Setup

This pipeline uses **three separate environments** due to PyTorch version conflicts between SAM2 and 4D Gaussian Splatting.

## 1. SAM2 Environment (`sam2`)

Used for: video segmentation / object masking (stages 5–7)

```bash
conda env create -f environments/sam2_environment.yml
conda activate sam2
cd sam2
pip install -e .
```

**Checkpoint download**: Download `sam2.1_hiera_large.pt` from [Meta's SAM2 releases](https://github.com/facebookresearch/sam2) and place in `sam2/checkpoints/`.

## 2. 4D Gaussian Splatting Environment (`4dgs`)

Used for: 4DGS training and point cloud export (stages 9–10)

```bash
conda env create -f environments/4dgs_environment.yml
conda activate 4dgs
cd <path-to-4DGaussians>/submodules/depth-diff-gaussian-rasterization
pip install -e .
cd ../simple-knn
pip install -e .
```

> **Note**: 4DGaussians must be cloned separately (has CUDA submodules that compile in-place). Set the path in `configs/paths.json`.

## 3. Stress Analysis Environment (`stress`)

Used for: mesh editing, cluster export, stress/displacement analysis (stages 11–15)

```bash
conda create -n stress python=3.11
conda activate stress
pip install -r environments/stress_requirements.txt
```

## Tested Configuration

| Component | Version |
|-----------|---------|
| OS | Windows 11 |
| CUDA | 12.4 |
| Python | 3.11 |
| PyTorch (SAM2) | 2.5.1+cu124 |
| PyTorch (4DGS) | 2.6.0+cu124 |
| GPU | NVIDIA RTX 4070 Ti (12 GB VRAM) |
| FFmpeg | Required on PATH |
