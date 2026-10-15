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
SAMPLE_DIR = PROJECT_ROOT / "sample"          # 训练样本输出目录
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"  # 模型权重保存目录
LOGS_DIR = PROJECT_ROOT / "logs"              # 训练日志目录
OUTPUT_DIR = PROJECT_ROOT / "output"          # 推理结果输出目录
DATASET_DIR = PROJECT_ROOT / "Dataset"        # 原始 CSV 数据目录
FACTORS_DIR = PROJECT_ROOT / "Controlling_Factors" / "feature"  # 13 个因子 TIF 目录

# -----------------------------------------------------------------------------
# 13 个控制因子栅格路径（顺序必须与模型输入顺序一致）
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
# 表格数据文件
# -----------------------------------------------------------------------------
LS_CSV = DATASET_DIR / "wuping_landslide_point_area_elev_stats.csv"

# -----------------------------------------------------------------------------
# 归一化统计量（基于完整样本集计算得到）
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
# 训练 / 数据常量
# -----------------------------------------------------------------------------
SEED = 42
PATCH_SIZE = 31
STRIDE = 2
BATCH_SIZE = 256
NUM_WORKERS = 0  # Windows 下的默认值；在 Linux/macOS 上若稳定可增大
MIN_BUFFER_DIST = 500  # 单位：米，非滑坡点采样时与滑坡点的最小距离
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
