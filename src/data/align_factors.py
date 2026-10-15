# -*- coding: utf-8 -*-
"""
将永川研究区的因子栅格对齐到统一网格。

背景：Dataset/yonchuan/factor 下的 TIF 来自不同处理流程，网格不一致：
  - 降雨组（Acc_* / Daily_Max / Max_3d_sliding）：645x756，精确 30m 像元
  - 地形组（Elevation 等）：668x782，约 29.59x29.44m
  - TRI/TPI/TWI、NDVI、Distanc_road 又是各自不同的网格
训练/推理流程要求所有因子严格对齐，因此全部重投影到参考网格。

参考网格选用降雨组的 30m 网格（像元为整数 30m，且后续添加降雨因子时
无需再对齐）。分类因子（岩性、地貌）用最近邻，连续因子用双线性。
所有输出的 nodata 统一为 NaN（float32）。

用法：
    # 永川（默认，行为不变）
    python src/data/align_factors.py

    # 新区：因子放入新区目录。源文件名优先匹配 FACTORS 表第一列，
    # 找不到时回退匹配标准名（即新区可直接用 Elevation.tif 等标准命名）
    python src/data/align_factors.py \
        --src Dataset/<新区>/factor --dst Dataset/<新区>/factor_aligned \
        --ref Dataset/<新区>/factor/<该区任一30m栅格.tif>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

# Allow imports from the project root when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import PROJECT_ROOT

# ========== 配置 ==========
SRC_DIR = PROJECT_ROOT / "Dataset" / "yonchuan" / "factor"
DST_DIR = PROJECT_ROOT / "Dataset" / "yonchuan" / "factor_aligned"

# 参考网格：降雨组任一栅格（30m 整像元）
REFERENCE = SRC_DIR / "Acc_3d_mm_30m.tif"

# (源文件名, 输出标准名, 是否分类数据, 数值缩放, 小于等于该值视为 nodata)
# 输出标准名与武平数据集保持一致，config.FACTOR_PATHS 只需换目录
# 已知数据问题：
#   - Elevation 边缘有 7765 个 0 值（nodata 被写成 0），需掩膜
#   - Distanc_road 的距离单位是度（在地理坐标系下计算后投影），乘 111000 近似转米
FACTORS = [
    ("Elevation.tif",                      "Elevation.tif",          False, 1.0,     0.0),
    ("Slope.tif",                          "Slope.tif",              False, 1.0,     None),
    ("Aspect.tif",                         "Aspect.tif",             False, 1.0,     None),
    ("Topographic Position Index.tif",     "TPI.tif",                False, 1.0,     None),
    ("Landform_class.tif",                 "Landform class.tif",     True,  1.0,     None),
    ("Terrain Ruggedness Index (TRI).tif", "TRI.tif",                False, 1.0,     None),
    ("Profile_Curvature.tif",              "Profile Curvature.tif",  False, 1.0,     None),
    ("Plan_Curvature.tif",                 "Plan Curvature.tif",     False, 1.0,     None),
    ("Topographic Wetness Index.tif",      "TWI.tif",                False, 1.0,     None),
    ("lithology.tif",                      "Lithology.tif",          True,  1.0,     None),
    ("Distanc_fault.tif",                  "Distance2fault.tif",     False, 1.0,     None),
    ("NDVI.tif",                           "NDVI.tif",               False, 1.0,     None),
    ("Distanc_road_ProjectRaste.tif",      "Distance2road.tif",      False, 111000.0, None),
    # 降雨因子（本身就在参考网格上，重投影为恒等变换）
    ("Daily_Max_mm_30m.tif",               "DailyMax.tif",           False, 1.0,     None),
    ("Acc_3d_mm_30m.tif",                  "Acc3d.tif",              False, 1.0,     None),
    ("Acc_7d_mm_30m.tif",                  "Acc7d.tif",              False, 1.0,     None),
]


def align_one(src_path, dst_path, ref_meta, is_categorical, scale, mask_leq):
    """把单个栅格重投影到参考网格，nodata 统一为 NaN。"""
    with rasterio.open(src_path) as src:
        src_arr = src.read(1)
        src_nodata = src.nodata

        src_arr = src_arr.astype(np.float32)
        # 源 nodata 转成 NaN，避免重投影时参与插值
        if src_nodata is not None and not (isinstance(src_nodata, float) and np.isnan(src_nodata)):
            src_arr = np.where(src_arr == src_nodata, np.nan, src_arr)
        if mask_leq is not None:
            src_arr = np.where(src_arr <= mask_leq, np.nan, src_arr)

        dst_arr = np.full((ref_meta["height"], ref_meta["width"]), np.nan, dtype=np.float32)
        reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=ref_meta["transform"],
            dst_crs=ref_meta["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.nearest if is_categorical else Resampling.bilinear,
        )

    if scale != 1.0:
        dst_arr = dst_arr * np.float32(scale)  # NaN 经乘法仍是 NaN

    profile = {
        "driver": "GTiff",
        "height": ref_meta["height"],
        "width": ref_meta["width"],
        "count": 1,
        "dtype": "float32",
        "crs": ref_meta["crs"],
        "transform": ref_meta["transform"],
        "compress": "lzw",
        "nodata": np.nan,
    }
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(dst_arr, 1)

    valid = np.isfinite(dst_arr)
    return valid.sum(), valid.size


def main():
    ap = argparse.ArgumentParser(description="因子栅格对齐到统一参考网格（nodata 统一为 NaN）")
    ap.add_argument("--src", default=str(SRC_DIR), help="原始因子目录")
    ap.add_argument("--dst", default=str(DST_DIR), help="对齐后输出目录")
    ap.add_argument("--ref", default=str(REFERENCE),
                    help="参考网格栅格（新区选该区任一 30m 栅格，建议 DEM 或降雨）")
    args = ap.parse_args()
    src_dir, dst_dir = Path(args.src), Path(args.dst)
    dst_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.ref) as ref:
        ref_meta = {
            "height": ref.height,
            "width": ref.width,
            "crs": ref.crs,
            "transform": ref.transform,
        }
    print("参考网格: %dx%d, res=%s, crs=%s" % (
        ref_meta["width"], ref_meta["height"],
        (ref_meta["transform"].a, abs(ref_meta["transform"].e)), ref_meta["crs"]))

    for src_name, dst_name, is_cat, scale, mask_leq in FACTORS:
        # 优先匹配原始文件名（应用表中针对永川数据的修正）；
        # 回退匹配标准名——标准命名的文件视为已清洗（单位为米、nodata 规范），不做修正
        src_path = src_dir / src_name
        fix_scale, fix_mask = scale, mask_leq
        if not src_path.exists():
            src_path = src_dir / dst_name
            fix_scale, fix_mask = 1.0, None
        dst_path = dst_dir / dst_name
        if not src_path.exists():
            print("  !! 缺失: %s（也未找到标准名 %s）" % (src_name, dst_name))
            continue
        n_valid, n_total = align_one(src_path, dst_path, ref_meta, is_cat, fix_scale, fix_mask)
        print("  %-30s -> %-25s 有效像元 %d/%d (%.1f%%) %s" % (
            src_path.name, dst_name, n_valid, n_total,
            100.0 * n_valid / n_total, "[分类]" if is_cat else ""))

    print("\n完成，输出目录: %s" % dst_dir)


if __name__ == "__main__":
    main()
