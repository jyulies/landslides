# -*- coding: utf-8 -*-
"""
从滑坡面 shapefile 采样正样本点并提取 31x31 patch 训练样本

输入：
  - 滑坡面 shapefile（面内多点采样；同时栅格化为掩膜供非滑坡点避让）
  - 16 个因子 TIF（已对齐，见 align_factors.py）
  - 非滑坡点：在全图范围内随机采样，排除滑坡面覆盖的像元
    （可用 NONLS_BUFFER_DIST 额外外扩缓冲，默认 0 = 仅避开面本身）

正样本采样策略（面内多点，替代旧的"每面仅质心 1 点"）：
  - 每个多边形取"像元中心严格落在面内"的像元（all_touched=False），
    上限 LS_MAX_POINTS_PER_POLY 个，贪心最远点选取得尽量分散
  - 内部像元为 0 的多边形（小面，永川约 1/4）回退到质心
  - 每个点附带 jitter 半径：距离变换给出的"偏移后中心仍在面内"的安全半径，
    训练时 RasterPatchDataset 每个 epoch 在半径内随机偏移中心（数据增强）
  - train/val 按多边形划分（同一面的所有点进同一侧），避免空间泄漏

输出：
  - sample/*.npy (C,31,31) float32 patch（SAVE_NPY=True 时；C = len(config.FACTOR_PATHS)）
  - train/val_labels.json、train/val_points.json（含 x, y, label, pid, j）

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
from pathlib import Path

import numpy as np
import rasterio
import shapefile  # pyshp
from rasterio.features import rasterize
from rasterio.transform import rowcol, xy
from scipy import ndimage
from tqdm import tqdm

# Allow imports from the project root when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import (
    FACTOR_PATHS, LS_SHP, SAMPLE_DIR, SEED,
    PATCH_SIZE, NONLS_BUFFER_DIST, N_NONLS_RATIO,
    LS_MAX_POINTS_PER_POLY, LS_MAX_JITTER_PX, LS_MIN_JITTER_PX
)

# ========== 配置 ==========
OUTPUT_DIR = str(SAMPLE_DIR)
PATCH = PATCH_SIZE
PAD = PATCH // 2  # 15

# False 时不写 .npy，只保存采样点 JSON（配合 RasterPatchDataset 内存直读模式，
# 适合大内存服务器；True 保持原有 .npy 流程）
SAVE_NPY = False


def read_all_factors():
    """读取所有因子到内存 (C, H, W)"""
    bands = []
    for fp in FACTOR_PATHS:
        with rasterio.open(fp) as src:
            bands.append(src.read(1).astype(np.float32))
    data = np.stack(bands, axis=0)
    return data


def coord_to_rowcol(transform, x, y):
    """投影坐标 -> 行列号"""
    r, c = rowcol(transform, x, y)
    return int(r), int(c)


def extract_patch(data, row, col, pad=PAD):
    """提取以(row,col)为中心的(C,31,31) patch，边界零填充"""
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


def _polygon_centroid(shp):
    """pyshp Shape -> 面积加权质心 (x, y)，支持多部件多边形"""
    pts = shp.points
    parts = list(shp.parts) + [len(pts)]
    total_a, cx_sum, cy_sum = 0.0, 0.0, 0.0
    for i in range(len(parts) - 1):
        ring = pts[parts[i]:parts[i + 1]]
        x = np.array([p[0] for p in ring])
        y = np.array([p[1] for p in ring])
        cross = x * np.roll(y, -1) - np.roll(x, -1) * y
        a = cross.sum() / 2.0
        if abs(a) < 1e-12:
            continue
        cx = ((x + np.roll(x, -1)) * cross).sum() / (6 * a)
        cy = ((y + np.roll(y, -1)) * cross).sum() / (6 * a)
        total_a += a
        cx_sum += cx * a
        cy_sum += cy * a
    if abs(total_a) < 1e-12:
        return pts[0]
    return cx_sum / total_a, cy_sum / total_a


def generate_ls_points_from_shp(shp_path, transform, H, W,
                                max_per_poly=LS_MAX_POINTS_PER_POLY,
                                max_jitter=LS_MAX_JITTER_PX):
    """
    每个滑坡多边形内部采样多个正样本点。

    返回 list of (pid, x, y, jitter_px)：
      - 候选 = 像元中心严格落在面内的像元（all_touched=False），
        取 min(候选数, max_per_poly) 个，贪心最远点选取得尽量分散
      - 无内部像元的多边形回退到质心（jitter=0）
      - jitter_px：距离变换给出的安全偏移半径（保守按对角线距离折算，
        保证偏移 sqrt(2)*j 后中心仍在面内），训练期随机抖动用；
        保底 LS_MIN_JITTER_PX 像元（编目边界本身有 ±1 像元量级误差）
    """
    sf = shapefile.Reader(str(shp_path))
    shapes = sf.shapes()
    geoms = [(shp.__geo_interface__, i) for i, shp in enumerate(shapes)]
    idmap = rasterize(
        geoms, out_shape=(H, W), transform=transform, fill=-1,
        all_touched=False, dtype="int32",
    )

    points = []
    n_fallback = 0
    for pid, shp in enumerate(tqdm(shapes, desc="Sampling polygons")):
        mask = idmap == pid
        npx = int(mask.sum())
        if npx == 0:
            cx, cy = _polygon_centroid(shp)
            points.append((pid, float(cx), float(cy), LS_MIN_JITTER_PX))
            n_fallback += 1
            continue

        dt = ndimage.distance_transform_edt(mask)  # 各内部像元到面外的距离（像元）
        cand = np.argwhere(mask).astype(np.float64)
        k = min(npx, max_per_poly)

        # 贪心最远点：首个取离边界最远的（最稳妥），之后逐个取离已选集合最远的
        first = int(np.argmax(dt[mask]))
        chosen = [first]
        while len(chosen) < k:
            d = np.min(
                np.linalg.norm(cand[:, None, :] - cand[None, np.array(chosen), :], axis=2),
                axis=1,
            )
            chosen.append(int(np.argmax(d)))

        for ci in chosen:
            r, c = int(cand[ci, 0]), int(cand[ci, 1])
            j = int(min(max((dt[r, c] - 0.5) / np.sqrt(2), 0.0), max_jitter))
            j = max(j, LS_MIN_JITTER_PX)
            x, y = xy(transform, r, c)
            points.append((pid, float(x), float(y), j))

    print("  多边形 %d 个 -> 正样本点 %d 个（%d 个小面回退质心）" % (
        len(shapes), len(points), n_fallback))
    return points


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

    # 2. 滑坡面内多点采样（每个多边形最多 LS_MAX_POINTS_PER_POLY 个点）
    print("\nStep 2: Sampling positive points inside landslide polygons (max %d/poly)..."
          % LS_MAX_POINTS_PER_POLY)
    ls_points = generate_ls_points_from_shp(LS_SHP, transform, H, W)  # (pid, x, y, j)

    # 3. 生成非滑坡点（避开滑坡面掩膜）
    n_target = int(len(ls_points) * N_NONLS_RATIO)
    print("\nStep 3: Generating non-landslide points (avoid landslide polygons, buffer=%d m)..." % NONLS_BUFFER_DIST)
    ls_mask = build_ls_mask(LS_SHP, transform, H, W, buffer_dist=NONLS_BUFFER_DIST)
    nonls_coords = generate_nonls_points(transform, H, W, ls_mask, n_target, seed=SEED)
    print("  Generated non-landslide points: %d" % len(nonls_coords))

    # 4. 提取patch
    print("\nStep 4: Extracting patches...")
    all_samples = []   # (name, label, x, y, pid, j)
    center_vals = {}   # name -> patch 中心像元各通道值（供 Step 6 验证用）

    # 清空旧的sample目录中的npy文件
    if SAVE_NPY:
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith('.npy'):
                os.remove(os.path.join(OUTPUT_DIR, f))

    # 滑坡 (label=1)
    skipped_ls = 0
    skipped_ls_nan = 0
    valid_ls = 0
    for i, (pid, x, y, j) in enumerate(tqdm(ls_points, desc="Landslide")):
        r, c = coord_to_rowcol(transform, x, y)
        if PAD <= r < H - PAD and PAD <= c < W - PAD:
            patch = extract_patch(data, r, c)
            if not np.isfinite(patch).all():
                skipped_ls_nan += 1
                continue
            name = "ls_%05d_%d" % (pid, i)
            if SAVE_NPY:
                np.save(os.path.join(OUTPUT_DIR, name + ".npy"), patch)
            center_vals[name] = patch[:, PAD, PAD].copy()
            all_samples.append((name, 1, x, y, pid, j))
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
            if SAVE_NPY:
                np.save(os.path.join(OUTPUT_DIR, name + ".npy"), patch)
            center_vals[name] = patch[:, PAD, PAD].copy()
            all_samples.append((name, 0, x, y, -1, 0))
            valid_non += 1
        else:
            skipped_non += 1

    print("\n  Valid: %d landslide + %d non-landslide" % (valid_ls, valid_non))
    if skipped_ls > 0 or skipped_non > 0:
        print("  Skipped (out of bounds): %d LS, %d non-LS" % (skipped_ls, skipped_non))
    if skipped_ls_nan > 0 or skipped_non_nan > 0:
        print("  Skipped (contains NaN): %d LS, %d non-LS" % (skipped_ls_nan, skipped_non_nan))

    # 5. 划分训练集/验证集：正样本按多边形划分（同一面的所有点进同一侧，
    #    防止空间泄漏）；非滑坡点随机划分
    print("\nStep 5: Splitting train/val (8:2, by polygon for LS)...")
    rng = np.random.RandomState(SEED)
    ls_pids = np.array(sorted({s[4] for s in all_samples if s[1] == 1}))
    rng.shuffle(ls_pids)
    n_val_pid = int(round(len(ls_pids) * 0.2))
    val_pids = set(ls_pids[:n_val_pid].tolist())

    train_dict, val_dict = {}, {}
    for name, label, x, y, pid, j in all_samples:
        if label == 1:
            (val_dict if pid in val_pids else train_dict)[name] = label
        # 非滑坡点稍后统一随机分

    nonls_names = [s[0] for s in all_samples if s[1] == 0]
    rng.shuffle(nonls_names)
    n_val_non = int(round(len(nonls_names) * 0.2))
    for i, name in enumerate(nonls_names):
        (val_dict if i < n_val_non else train_dict)[name] = 0

    with open(os.path.join(OUTPUT_DIR, "train_labels.json"), "w") as f:
        json.dump(train_dict, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "val_labels.json"), "w") as f:
        json.dump(val_dict, f, indent=2)

    # 采样点 JSON（内存直读模式 RasterPatchDataset 使用；j = 训练期安全抖动半径）
    rec_by_name = {s[0]: s for s in all_samples}
    def _points_records(d):
        recs = []
        for n in d:
            _, label, x, y, pid, j = rec_by_name[n]
            recs.append({"name": n, "x": x, "y": y, "label": label, "pid": pid, "j": j})
        return recs
    with open(os.path.join(OUTPUT_DIR, "train_points.json"), "w") as f:
        json.dump(_points_records(train_dict), f)
    with open(os.path.join(OUTPUT_DIR, "val_points.json"), "w") as f:
        json.dump(_points_records(val_dict), f)

    # 6. 验证数据质量
    print("\nStep 6: Validating data quality...")
    n_check = min(2000, len(train_dict))  # 增加检查样本数，结果更稳定
    ls_names = [k for k, v in train_dict.items() if v == 1]
    nols_names = [k for k, v in train_dict.items() if v == 0]
    ls_check = rng.choice(ls_names, size=min(n_check, len(ls_names)), replace=False).tolist()
    nols_check = rng.choice(nols_names, size=min(n_check, len(nols_names)), replace=False).tolist()

    ls_vals = np.array([center_vals[n] for n in ls_check])
    nols_vals = np.array([center_vals[n] for n in nols_check])

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

    n_train_ls = sum(1 for v in train_dict.values() if v == 1)
    n_val_ls = sum(1 for v in val_dict.values() if v == 1)
    print("\n" + "=" * 50)
    print("Done!")
    print("  Total: %d" % len(all_samples))
    print("  Train: %d (LS=%d, non=%d)" % (len(train_dict), n_train_ls, len(train_dict) - n_train_ls))
    print("  Val:   %d (LS=%d, non=%d)" % (len(val_dict), n_val_ls, len(val_dict) - n_val_ls))
    print("  Output: %s" % OUTPUT_DIR)
    print("=" * 50)


if __name__ == "__main__":
    main()
