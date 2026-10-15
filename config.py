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
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# 项目根目录：默认取 config.py 所在目录（即克隆位置），开箱即用；
# 需要覆盖时设置 LANDSLIDE_ROOT 环境变量
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("LANDSLIDE_ROOT", Path(__file__).resolve().parent))

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
# 归一化统计量（基于永川 8021 个 16 通道样本 patch 的所有像元计算；
# 正样本 = 滑坡面内多点采样，见 extract_patches.py）
# 通道顺序 = FACTOR_PATHS：13 静态因子 + DailyMax/Acc3d/Acc7d
# -----------------------------------------------------------------------------
MEAN = [
    480.739376, 16.923965, 190.493486, 0.014844, 6.059110,
    6.444507, -0.000248, 0.000950, 7.052289, 4.051166,
    2870.680181, 93.692619, 318.930814, 62.657755, 105.427465,
    123.571305,
]
STD = [
    171.721294, 10.010436, 92.120911, 6.264623, 1.818615,
    3.781557, 0.003240, 0.060249, 1.733574, 1.635219,
    2231.025014, 5.071201, 273.013350, 5.783752, 8.084047,
    8.928690,
]

# -----------------------------------------------------------------------------
# 训练 / 数据常量
# -----------------------------------------------------------------------------
SEED = 42
PATCH_SIZE = 31
STRIDE = 2
BATCH_SIZE = 256
# Windows 下 DataLoader 多进程需 spawn（Optuna 路径会崩），用 0；
# Linux/macOS 用 fork，可安全开大
NUM_WORKERS = 0 if sys.platform == "win32" else 8
NONLS_BUFFER_DIST = 0  # 单位：米，非滑坡点与滑坡面边界的最小距离；0 = 仅避开面本身
N_NONLS_RATIO = 1      # 非滑坡点数量 = 滑坡点数量 × 该比例

# 面内多点采样（extract_patches.py）：每个滑坡多边形最多取多少个内部像元作为正样本；
# 永川 97% 的多边形 ≤ 5000 m²（约 5 个像元），上限 5 足够
LS_MAX_POINTS_PER_POLY = 5
# 训练期中心抖动上限（像元）：采样时用距离变换记录每个点可安全偏移的半径
# （偏移后中心仍落在滑坡面内），训练时每个 epoch 在半径内随机偏移
LS_MAX_JITTER_PX = 4
# 所有正样本的保底抖动半径（像元）。永川滑坡面中位数仅 ~1 像元，
# 严格"面内"抖动对绝大多数面为 0；编目边界本身有 ±1 像元（30m）量级的
# 定位/制图误差，±1 像元的偏移在误差范围内，可安全用于全部正样本
LS_MIN_JITTER_PX = 1

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
