# -*- coding: utf-8 -*-
"""
landslide 易发性制图项目的集中配置。

路径由 PROJECT_ROOT 派生；PROJECT_ROOT 优先读取 LANDSLIDE_ROOT 环境变量，
否则回退到默认项目位置。

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
# 项目根目录
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("LANDSLIDE_ROOT", r"C:\code\landslides"))

# -----------------------------------------------------------------------------
# 常用目录
# -----------------------------------------------------------------------------
SAMPLE_DIR = PROJECT_ROOT / "sample_yongchuan"  # 训练样本输出目录（永川，避免覆盖武平样本）
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"  # 模型权重保存目录
LOGS_DIR = PROJECT_ROOT / "logs"              # 训练日志目录
OUTPUT_DIR = PROJECT_ROOT / "output"          # 推理结果输出目录
DATASET_DIR = PROJECT_ROOT / "Dataset" / "yonchuan"  # 永川原始数据目录
FACTORS_DIR = DATASET_DIR / "factor_aligned"  # 因子 TIF 目录（已对齐到 30m 网格）

# -----------------------------------------------------------------------------
# 16 个控制因子栅格路径（顺序必须与模型输入顺序一致）
# 前 13 个为静态因子，后 3 个为降雨因子（最大单日、3 日累计、7 日累计）
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
    FACTORS_DIR / "DailyMax.tif",
    FACTORS_DIR / "Acc3d.tif",
    FACTORS_DIR / "Acc7d.tif",
]

# 模型输入通道数（全项目统一以此为准，不要硬编码）
IN_CH = len(FACTOR_PATHS)

# -----------------------------------------------------------------------------
# 表格数据文件（由 src/data/shp_centroids_to_csv.py 从滑坡多边形生成）
# -----------------------------------------------------------------------------
LS_CSV = DATASET_DIR / "yongchuan_landslide_centroids.csv"
LS_SHP = DATASET_DIR / "shp" / "landslides.shp"  # 滑坡面 shapefile（非滑坡点采样时避开）

# -----------------------------------------------------------------------------
# 归一化统计量（基于永川全部 4738 个 16 通道样本 patch 的所有像元计算）
# 通道顺序 = FACTOR_PATHS：13 静态因子 + DailyMax/Acc3d/Acc7d
# -----------------------------------------------------------------------------
MEAN = [
    481.575613, 16.792950, 189.711012, 0.026458, 6.052846,
    6.391150, -0.000237, 0.000973, 7.055495, 4.075471,
    2903.992182, 93.710679, 316.825109, 62.555157, 105.280130,
    123.420660,
]
STD = [
    171.983333, 9.917636, 91.620365, 6.211514, 1.817832,
    3.743806, 0.003244, 0.059943, 1.720111, 1.632523,
    2217.522557, 5.037115, 272.765130, 5.699816, 7.947572,
    8.790169,
]

# -----------------------------------------------------------------------------
# 训练 / 数据常量
# -----------------------------------------------------------------------------
SEED = 42
PATCH_SIZE = 31
STRIDE = 2
BATCH_SIZE = 256
NUM_WORKERS = 0  # Windows 下的默认值；在 Linux/macOS 上若稳定可增大
NONLS_BUFFER_DIST = 0  # 单位：米，非滑坡点与滑坡面边界的最小距离；0 = 仅避开面本身
N_NONLS_RATIO = 1      # 非滑坡点数量 = 滑坡点数量 × 该比例

# -----------------------------------------------------------------------------
# 便捷工具函数
# -----------------------------------------------------------------------------
def ensure_dirs():
    """若输出 / 日志目录不存在则创建。"""
    for d in (SAMPLE_DIR, CHECKPOINTS_DIR, LOGS_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def factor_paths_str() -> list:
    """返回字符串形式的因子路径，兼容旧版 rasterio 用法。"""
    return [str(p) for p in FACTOR_PATHS]
