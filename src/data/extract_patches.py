# -*- coding: utf-8 -*-
"""
从CSV坐标文件提取31x31 patch训练样本

输入：
  - 滑坡点CSV (x, y投影坐标，由 shp_centroids_to_csv.py 从滑坡面生成)
  - 滑坡面 shapefile（栅格化为掩膜，非滑坡点采样时避开）
  - 13个因子TIF（已对齐，见 align_factors.py）
  - 非滑坡点：在全图范围内随机采样，排除滑坡面覆盖的像元
    （可用 NONLS_BUFFER_DIST 额外外扩缓冲，默认 0 = 仅避开面本身）
输出：
  - sample/*.npy (C,31,31) float32 patch（C = len(config.FACTOR_PATHS)）
  - train_labels.json / val_labels.json

用法：
    python src/data/extract_patches.py

Reference / 引用：
    Luo, S., Mao, W., Yang, Z., Zheng, G., He, Z., Wang, J., & Huang, Y. (2026).
    CNXT-Ti-LT--Based Multi-Scale Feature--Aware Susceptibility Mapping of
    Rainfall-Induced Clustered Landslides in Southeast China.
    Journal of Geophysical Research: Machine Learning and Computation,
    3, e2025JH001115. https://doi.org/10.1029/2025JH001115

    Training dataset / 训练样本与因子: https://doi.org/10.5281/zenodo.18463063
    Statistical data / 统计表格数据: https://doi.org/10.5281/zenodo.18226013
"""

import os
import sys
import json
import csv
from pathlib import Path

import numpy as np
import rasterio
import shapefile  # pyshp
from rasterio.features import rasterize
from rasterio.transform import rowcol, xy
from scipy import ndimage
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Allow imports from the project root when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import (
    FACTOR_PATHS, LS_CSV, LS_SHP, SAMPLE_DIR, SEED,
    PATCH_SIZE, NONLS_BUFFER_DIST, N_NONLS_RATIO
)

# ========== 配置 ==========
OUTPUT_DIR = str(SAMPLE_DIR)
PATCH = PATCH_SIZE
PAD = PATCH // 2  # 15


def read_all_factors():
    """读取所有因子到内存 (13, H, W)"""
    bands = []
    for fp in FACTOR_PATHS:
        with rasterio.open(fp) as src:
            bands.append(src.read(1).astype(np.float32))
    data = np.stack(bands, axis=0)
    return data


def read_ls_points_from_csv(csv_path):
    """读取滑坡点CSV中的x,y坐标"""
    coords = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                x = float(row['x'].strip())
                y = float(row['y'].strip())
                coords.append((x, y))
            except (KeyError, ValueError) as e:
                if i < 5:
                    print("  跳过行 %d: %s" % (i, e))
                continue
    return coords


def coord_to_rowcol(transform, x, y):
    """投影坐标 -> 行列号"""
    r, c = rowcol(transform, x, y)
    return int(r), int(c)


def extract_patch(data, row, col, pad=PAD):
    """提取以(row,col)为中心的(13,31,31) patch，边界零填充"""
    C, H, W = data.shape
    r0, r1 = row - pad, row + pad + 1
    c0, c1 = col - pad, col + pad + 1

    patch = np.zeros((C, PATCH, PATCH), dtype=np.float32)

    pr0, pr1 = max(0, r0), min(H, r1)
    pc0, pc1 = max(0, c0), min(W, c1)
    ppr0 = pr0 - r0
    ppr1 = ppr0 + (pr1 - pr0)
    ppc0 = pc0 - c0
    ppc1 = ppc0 + (c1 - c0)

    patch[:, ppr0:ppr1, ppc0:ppc1] = data[:, pr0:pr1, pc0:pc1]
    return patch


def build_ls_mask(shp_path, transform, H, W, buffer_dist=0):
    """
    把滑坡面 shapefile 栅格化为布尔掩膜 (H, W)，True = 滑坡面覆盖的像元。

    all_touched=True：只要像元被多边形碰到就算滑坡，宁可多排除。
    buffer_dist > 0 时按米外扩（换算成像元做二值膨胀）。
    """
    sf = shapefile.Reader(str(shp_path))
    geoms = [shp.__geo_interface__ for shp in sf.shapes()]
    mask = rasterize(
        geoms,
        out_shape=(H, W),
        transform=transform,
        fill=0,
        default_value=1,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)

    n_poly_px = int(mask.sum())
    if buffer_dist > 0:
        res = abs(transform.a)  # 像元大小（米）
        n_iter = int(np.ceil(buffer_dist / res))
        mask = ndimage.binary_dilation(mask, iterations=n_iter)
        print("  滑坡面 %d 像元，外扩 %d m (%d 像元) 后 %d 像元" % (
            n_poly_px, buffer_dist, n_iter, int(mask.sum())))
    else:
        print("  滑坡面掩膜: %d 像元 (%.2f%% of 全图)" % (
            n_poly_px, 100.0 * n_poly_px / (H * W)))
    return mask


def generate_nonls_points(transform, H, W, ls_mask, n_target, seed=42):
    """
    在TIF范围内随机生成非滑坡点：候选像元需未被滑坡面掩膜覆盖，
    且距图幅边界 >= PAD（保证能取出完整 patch）。
    """
    rng = np.random.RandomState(seed)

    # 候选像元：掩膜外 + 边界内
    candidates = np.argwhere(~ls_mask)
    candidates = candidates[
        (candidates[:, 0] >= PAD) & (candidates[:, 0] < H - PAD) &
        (candidates[:, 1] >= PAD) & (candidates[:, 1] < W - PAD)
    ]
    print("  候选非滑坡像元: %d" % len(candidates))

    if len(candidates) < n_target:
        print("  !! 候选像元不足 %d，全部使用" % n_target)
        chosen = candidates
    else:
        idx = rng.choice(len(candidates), size=n_target, replace=False)
        chosen = candidates[idx]

    nonls_coords = []
    for r, c in chosen:
        x, y = xy(transform, int(r), int(c))
        nonls_coords.append((x, y))
    return nonls_coords


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(SEED)

    # 1. 读取因子
    print("=" * 50)
    print("Step 1: Reading %d factor TIFs..." % len(FACTOR_PATHS))
    data = read_all_factors()
    C, H, W = data.shape
    print("  Factor stack: %s" % str(data.shape))

    # 获取坐标系transform
    with rasterio.open(FACTOR_PATHS[0]) as src:
        transform = src.transform
        crs = src.crs
    print("  CRS: %s" % crs)

    # 2. 读取滑坡点
    print("\nStep 2: Reading landslide points...")
    ls_coords = read_ls_points_from_csv(LS_CSV)
    print("  Landslide points: %d" % len(ls_coords))

    # 3. 生成非滑坡点（避开滑坡面掩膜）
    n_target = int(len(ls_coords) * N_NONLS_RATIO)
    print("\nStep 3: Generating non-landslide points (avoid landslide polygons, buffer=%d m)..." % NONLS_BUFFER_DIST)
    ls_mask = build_ls_mask(LS_SHP, transform, H, W, buffer_dist=NONLS_BUFFER_DIST)
    nonls_coords = generate_nonls_points(transform, H, W, ls_mask, n_target, seed=SEED)
    print("  Generated non-landslide points: %d" % len(nonls_coords))

    # 4. 提取patch
    print("\nStep 4: Extracting patches...")
    all_samples = []

    # 清空旧的sample目录中的npy文件
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith('.npy'):
            os.remove(os.path.join(OUTPUT_DIR, f))

    # 滑坡 (label=1)
    skipped_ls = 0
    skipped_ls_nan = 0
    valid_ls = 0
    for i, (x, y) in enumerate(tqdm(ls_coords, desc="Landslide")):
        r, c = coord_to_rowcol(transform, x, y)
        if PAD <= r < H - PAD and PAD <= c < W - PAD:
            patch = extract_patch(data, r, c)
            if not np.isfinite(patch).all():
                skipped_ls_nan += 1
                continue
            name = "ls_%05d" % i
            np.save(os.path.join(OUTPUT_DIR, name + ".npy"), patch)
            all_samples.append((name, 1))
            valid_ls += 1
        else:
            skipped_ls += 1

    # 非滑坡 (label=0)
    skipped_non = 0
    skipped_non_nan = 0
    valid_non = 0
    for i, (x, y) in enumerate(tqdm(nonls_coords, desc="Non-landslide")):
        r, c = coord_to_rowcol(transform, x, y)
        if PAD <= r < H - PAD and PAD <= c < W - PAD:
            patch = extract_patch(data, r, c)
            if not np.isfinite(patch).all():
                skipped_non_nan += 1
                continue
            name = "nols_%05d" % i
            np.save(os.path.join(OUTPUT_DIR, name + ".npy"), patch)
            all_samples.append((name, 0))
            valid_non += 1
        else:
            skipped_non += 1

    print("\n  Valid: %d landslide + %d non-landslide" % (valid_ls, valid_non))
    if skipped_ls > 0 or skipped_non > 0:
        print("  Skipped (out of bounds): %d LS, %d non-LS" % (skipped_ls, skipped_non))
    if skipped_ls_nan > 0 or skipped_non_nan > 0:
        print("  Skipped (contains NaN): %d LS, %d non-LS" % (skipped_ls_nan, skipped_non_nan))

    # 5. 划分训练集/验证集
    print("\nStep 5: Splitting train/val (8:2)...")
    names = [s[0] for s in all_samples]
    labels = [s[1] for s in all_samples]

    train_names, val_names, train_labels, val_labels = train_test_split(
        names, labels, test_size=0.2, random_state=SEED, stratify=labels
    )

    train_dict = {n: l for n, l in zip(train_names, train_labels)}
    val_dict = {n: l for n, l in zip(val_names, val_labels)}

    with open(os.path.join(OUTPUT_DIR, "train_labels.json"), "w") as f:
        json.dump(train_dict, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "val_labels.json"), "w") as f:
        json.dump(val_dict, f, indent=2)

    # 6. 验证数据质量
    print("\nStep 6: Validating data quality...")
    n_check = min(2000, len(train_dict))  # 增加检查样本数，结果更稳定
    rng = np.random.RandomState(SEED)
    ls_names = [k for k, v in train_dict.items() if v == 1]
    nols_names = [k for k, v in train_dict.items() if v == 0]
    ls_check = rng.choice(ls_names, size=min(n_check, len(ls_names)), replace=False).tolist()
    nols_check = rng.choice(nols_names, size=min(n_check, len(nols_names)), replace=False).tolist()

    def _center_values(names):
        """读取每个样本中心像元 (row=15, col=15) 的 13 通道数值。"""
        vals = []
        for name in names:
            arr = np.load(os.path.join(OUTPUT_DIR, name + ".npy")).astype(np.float32)
            vals.append(arr[:, PAD, PAD])
        return np.array(vals)

    ls_vals = _center_values(ls_check)
    nols_vals = _center_values(nols_check)

    ch_names = ['Elevation', 'Slope', 'Aspect', 'TPI', 'Landform', 'TRI',
                'ProfileCurv', 'PlanCurv', 'TWI', 'Lithology', 'Dist2fault', 'NDVI', 'Dist2road',
                'DailyMax', 'Acc3d', 'Acc7d']
    print("  Cohen's d (效应量: >0.8=大, 0.5=中, 0.2=小):")
    for i in range(len(ch_names)):
        ls_m = ls_vals[:, i].mean()
        nols_m = nols_vals[:, i].mean()
        ls_s = ls_vals[:, i].std(ddof=1)
        nols_s = nols_vals[:, i].std(ddof=1)
        n1, n2 = len(ls_vals), len(nols_vals)
        # 标准合并标准差（样本标准差）
        pooled = np.sqrt(((n1 - 1) * ls_s ** 2 + (n2 - 1) * nols_s ** 2) / (n1 + n2 - 2))
        d = abs(ls_m - nols_m) / pooled if pooled > 0 else 0
        level = 'LARGE' if d > 0.8 else ('MEDIUM' if d > 0.5 else ('SMALL' if d > 0.2 else 'tiny'))
        print("    Ch%2d %-15s: LS=%10.3f  nonLS=%10.3f  d=%.3f (%s)" % (i, ch_names[i], ls_m, nols_m, d, level))

    print("\n" + "=" * 50)
    print("Done!")
    print("  Total: %d" % len(all_samples))
    print("  Train: %d (LS=%d, non=%d)" % (len(train_dict), sum(train_labels), len(train_labels) - sum(train_labels)))
    print("  Val:   %d (LS=%d, non=%d)" % (len(val_dict), sum(val_labels), len(val_labels) - sum(val_labels)))
    print("  Output: %s" % OUTPUT_DIR)
    print("=" * 50)


if __name__ == "__main__":
    main()
