# -*- coding: utf-8 -*-
"""
Centralized configuration for the landslide susceptibility project.

Paths are derived from PROJECT_ROOT, which is read from the LANDSLIDE_ROOT
environment variable or falls back to the default project location.

Reference / 引用：
    Luo, S., Mao, W., Yang, Z., Zheng, G., He, Z., Wang, J., & Huang, Y. (2026).
    CNXT-Ti-LT--Based Multi-Scale Feature--Aware Susceptibility Mapping of
    Rainfall-Induced Clustered Landslides in Southeast China.
    Journal of Geophysical Research: Machine Learning and Computation,
    3, e2025JH001115. https://doi.org/10.1029/2025JH001115

    Training dataset / 训练样本与因子: https://doi.org/10.5281/zenodo.18463063
"""
import os
from pathlib import Path

# -----------------------------------------------------------------------------
# Project root
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("LANDSLIDE_ROOT", r"C:\code\landslides"))

# -----------------------------------------------------------------------------
# Common directories
# -----------------------------------------------------------------------------
SAMPLE_DIR = PROJECT_ROOT / "sample"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "output"
DATASET_DIR = PROJECT_ROOT / "Dataset"
FACTORS_DIR = PROJECT_ROOT / "Controlling_Factors" / "feature"

# -----------------------------------------------------------------------------
# 13 controlling-factor raster paths (order must match the model)
# -----------------------------------------------------------------------------
FACTOR_PATHS = [
    FACTORS_DIR / "Elevation.tif",
    FACTORS_DIR / "Slope.tif",
    FACTORS_DIR / "Aspect.tif",
    FACTORS_DIR / "TPI.tif",
    FACTORS_DIR / "Landform class.tif",
    FACTORS_DIR / "TRI.tif",
    FACTORS_DIR / "Profile Curvature.tif",
    FACTORS_DIR / "Plan Curvature.tif",
    FACTORS_DIR / "TWI.tif",
    FACTORS_DIR / "Lithology.tif",
    FACTORS_DIR / "Distance2fault.tif",
    FACTORS_DIR / "NDVI.tif",
    FACTORS_DIR / "Distance2road.tif",
]

# -----------------------------------------------------------------------------
# Tabular data files
# -----------------------------------------------------------------------------
LS_CSV = DATASET_DIR / "wuping_landslide_point_area_elev_stats.csv"

# -----------------------------------------------------------------------------
# Normalization statistics (computed over the full sample set)
# -----------------------------------------------------------------------------
MEAN = [
    417.616119, 17.790653, 187.580414, -0.012278, 5.720180,
    8.259916, -0.000228, 0.000378, 6.304661, 2.466416,
    2844.403564, 0.289612, 197.655365,
]
STD = [
    185.132370, 8.436256, 101.343697, 2.483558, 2.390866,
    3.502158, 0.003821, 0.024846, 3.022274, 1.545655,
    2511.757568, 0.090679, 196.766006,
]

# -----------------------------------------------------------------------------
# Training / data constants
# -----------------------------------------------------------------------------
SEED = 42
PATCH_SIZE = 31
STRIDE = 2
BATCH_SIZE = 256
NUM_WORKERS = 0  # default for Windows; increase on Linux/macOS if stable
MIN_BUFFER_DIST = 500  # meters, non-landslide point sampling
N_NONLS_RATIO = 1      # non-landslide count = landslide count * ratio

# -----------------------------------------------------------------------------
# Convenience helpers
# -----------------------------------------------------------------------------
def ensure_dirs():
    """Create output / log directories if they do not exist."""
    for d in (SAMPLE_DIR, CHECKPOINTS_DIR, LOGS_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def factor_paths_str() -> list:
    """Return factor paths as plain strings for legacy rasterio usage."""
    return [str(p) for p in FACTOR_PATHS]
