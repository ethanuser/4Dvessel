# AGENTS.md

This file provides guidance for AI coding agents and future automated edits to this repository.

## Project overview

This repository contains the codebase for the manuscript:

**4D Vessel Reconstruction for Benchtop Thrombectomy Analysis**

The pipeline supports multi-view video preprocessing, SAM2-based segmentation, 4D Gaussian Splatting reconstruction preparation, point-cloud/mesh processing, displacement analysis, and surface-based stress-proxy analysis for benchtop thrombectomy vessel experiments and synthetic validation experiments.

## High-level repository structure

```text
4Dvessel/
├── configs/                     # Path configs and experiment templates
├── environments/                # Conda/Python environment files
├── scripts/                     # Sequential PowerShell pipeline wrappers
├── docs/                        # Pipeline documentation and release checklists
├── Preprocess/                  # Calibration, video splitting, NeRF/4DGS preparation
├── sam2/                        # SAM2 segmentation utilities
├── vessel-stress-analysis/      # Mesh processing, displacement, stress-proxy analysis
└── vessel_blender_code/         # Synthetic validation and Blender scripts