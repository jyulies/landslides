# -*- coding: utf-8 -*-
"""
从CSV坐标文件提取31x31 patch训练样本

输入：
  - 滑坡点CSV (x, y投影坐标)
  - 13个因子TIF（已对齐）
  - 非滑坡点：在全图范围内随机采样，排除滑坡缓冲区(>=500m)
输出：
  - sample/*.npy (13,31,31) float32 patch
  - train_labels.json / val_labels.json

用法：
    python src/data/extract_patches.py
"""

import os
import sys
import json
import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import rowcol, xy
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Allow imports from the project root when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import (
    FACTOR_PATHS, LS_CSV, SAMPLE_DIR, SEED,
    PATCH_SIZE, MIN_BUFFER_DIST, N_NONLS_RATIO
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


def generate_nonls_points(transform, H, W, ls_coords, n_target, min_dist, seed=42):
    """
    在TIF范围内随机生成非滑坡点，确保每个点距所有滑坡点 >= min_dist 米。

    使用空间索引加速：先将滑坡点按网格分桶，只检查附近的滑坡点。
    """
    rng = np.random.RandomState(seed)

    # 滑坡点坐标数组
    ls_arr = np.array(ls_coords)  # (N, 2)

    # 计算TIF的投影坐标范围
    x_min, y_max = xy(transform, 0, 0)
    x_max, y_min = xy(transform, H - 1, W - 1)

    print("  TIF坐标范围: x=[%.1f, %.1f], y=[%.1f, %.1f]" % (x_min, x_max, y_min, y_max))
    print("  滑坡缓冲距离: %d m" % min_dist)

    # 网格分桶加速距离查询
    cell_size = min_dist
    n_cells_x = int(np.ceil((x_max - x_min) / cell_size))
    n_cells_y = int(np.ceil((y_max - y_min) / cell_size))

    # 将滑坡点放入网格
    ls_grid = {}
    for lx, ly in ls_coords:
        ci = int((lx - x_min) / cell_size)
        cj = int((ly - y_min) / cell_size)
        key = (ci, cj)
        if key not in ls_grid:
            ls_grid[key] = []
        ls_grid[key].append((lx, ly))

    # 搜索范围：需要检查周围 (2r+1)x(2r+1) 的网格
    r_cells = int(np.ceil(min_dist / cell_size)) + 1

    nonls_coords = []
    max_attempts = n_target * 200
    attempts = 0

    pbar = tqdm(total=n_target, desc="Generating non-LS points")

    while len(nonls_coords) < n_target and attempts < max_attempts:
        # 在TIF范围内随机采样一个像素坐标
        r = rng.randint(PAD, H - PAD)
        c = rng.randint(PAD, W - PAD)
        x, y = xy(transform, r, c)

        # 快速检查：计算所在网格，只检查周围网格内的滑坡点
        ci = int((x - x_min) / cell_size)
        cj = int((y - y_min) / cell_size)

        too_close = False
        for di in range(-r_cells, r_cells + 1):
            if too_close:
                break
            for dj in range(-r_cells, r_cells + 1):
                if too_close:
                    break
                key = (ci + di, cj + dj)
                if key in ls_grid:
                    for lx, ly in ls_grid[key]:
                        dist = np.sqrt((x - lx) ** 2 + (y - ly) ** 2)
                        if dist < min_dist:
                            too_close = True
                            break

        if not too_close:
            nonls_coords.append((x, y))
            pbar.update(1)

        attempts += 1
        if attempts % 10000 == 0:
            print("  Attempts: %d, Found: %d / %d" % (attempts, len(nonls_coords), n_target))

    pbar.close()
    return nonls_coords


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(SEED)

    # 1. 读取因子
    print("=" * 50)
    print("Step 1: Reading 13 factor TIFs...")
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

    # 3. 生成非滑坡点（远离滑坡 >= 500m）
    n_target = int(len(ls_coords) * N_NONLS_RATIO)
    print("\nStep 3: Generating non-landslide points (buffer >= %d m)..." % MIN_BUFFER_DIST)
    nonls_coords = generate_nonls_points(transform, H, W, ls_coords, n_target, MIN_BUFFER_DIST, seed=SEED)
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
    valid_ls = 0
    for i, (x, y) in enumerate(tqdm(ls_coords, desc="Landslide")):
        r, c = coord_to_rowcol(transform, x, y)
        if PAD <= r < H - PAD and PAD <= c < W - PAD:
            patch = extract_patch(data, r, c)
            name = "ls_%05d" % i
            np.save(os.path.join(OUTPUT_DIR, name + ".npy"), patch)
            all_samples.append((name, 1))
            valid_ls += 1
        else:
            skipped_ls += 1

    # 非滑坡 (label=0)
    skipped_non = 0
    valid_non = 0
    for i, (x, y) in enumerate(tqdm(nonls_coords, desc="Non-landslide")):
        r, c = coord_to_rowcol(transform, x, y)
        if PAD <= r < H - PAD and PAD <= c < W - PAD:
            patch = extract_patch(data, r, c)
            name = "nols_%05d" % i
            np.save(os.path.join(OUTPUT_DIR, name + ".npy"), patch)
            all_samples.append((name, 0))
            valid_non += 1
        else:
            skipped_non += 1

    print("\n  Valid: %d landslide + %d non-landslide" % (valid_ls, valid_non))
    if skipped_ls > 0 or skipped_non > 0:
        print("  Skipped (out of bounds): %d LS, %d non-LS" % (skipped_ls, skipped_non))

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
    n_check = min(200, len(train_dict))
    ls_check = [k for k, v in train_dict.items() if v == 1][:n_check]
    nols_check = [k for k, v in train_dict.items() if v == 0][:n_check]

    ls_means = []
    for name in ls_check:
        arr = np.load(os.path.join(OUTPUT_DIR, name + ".npy")).astype(np.float32)
        ls_means.append(arr.mean(axis=(1, 2)))
    ls_means = np.array(ls_means)

    nols_means = []
    for name in nols_check:
        arr = np.load(os.path.join(OUTPUT_DIR, name + ".npy")).astype(np.float32)
        nols_means.append(arr.mean(axis=(1, 2)))
    nols_means = np.array(nols_means)

    ch_names = ['Elevation', 'Slope', 'Aspect', 'TPI', 'Landform', 'TRI',
                'ProfileCurv', 'PlanCurv', 'TWI', 'Lithology', 'Dist2fault', 'NDVI', 'Dist2road']
    print("  Cohen's d (效应量: >0.8=大, 0.5=中, 0.2=小):")
    for i in range(13):
        ls_m = ls_means[:, i].mean()
        nols_m = nols_means[:, i].mean()
        pooled = (ls_means[:, i].std() + nols_means[:, i].std()) / 2
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
