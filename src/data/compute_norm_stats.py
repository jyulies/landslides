# -*- coding: utf-8 -*-
"""
重算多区域训练的通道归一化统计量（MEAN/STD）。

口径与永川原统计一致：对 config.TRAIN_REGIONS 各区的训练样本点
（sample_multi/<region>/train_points.json）切 31x31 patch，
累积所有 patch 所有像元的逐通道 sum/sumsq，最后算出 mean/std。

只统计有限值像元（NaN 不参与）；结果打印出来，手动填入 config.py。

用法：
    python src/data/compute_norm_stats.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import rowcol
from tqdm import tqdm

# Allow imports from the project root when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import (
    SAMPLE_MULTI_DIR, TRAIN_REGIONS, PATCH_SIZE, region_factor_paths,
)
from src.data.extract_patches import extract_patch

PAD = PATCH_SIZE // 2


def main():
    C = len(region_factor_paths(TRAIN_REGIONS[0]))
    sum_ = np.zeros(C, dtype=np.float64)
    sumsq = np.zeros(C, dtype=np.float64)
    cnt = np.zeros(C, dtype=np.int64)

    for region in TRAIN_REGIONS:
        pts_json = SAMPLE_MULTI_DIR / region / "train_points.json"
        with open(pts_json, encoding="utf-8") as f:
            pts = json.load(f)
        bands = []
        for fp in region_factor_paths(region):
            with rasterio.open(fp) as src:
                bands.append(src.read(1).astype(np.float32))
                transform = src.transform
        data = np.stack(bands, axis=0)
        assert data.shape[0] == C, "%s 通道数不一致" % region

        for p in tqdm(pts, desc=region):
            r, c = rowcol(transform, p["x"], p["y"])
            r, c = int(r), int(c)
            if not (PAD <= r < data.shape[1] - PAD and PAD <= c < data.shape[2] - PAD):
                continue
            patch = extract_patch(data, r, c)          # (C, P, P)
            flat = patch.reshape(C, -1)                # (C, P*P)
            valid = np.isfinite(flat)
            sum_ += np.where(valid, flat, 0).sum(axis=1)
            sumsq += np.where(valid, flat ** 2, 0).sum(axis=1)
            cnt += valid.sum(axis=1)

    mean = sum_ / cnt
    var = sumsq / cnt - mean ** 2
    std = np.sqrt(np.maximum(var, 0))

    names = [p.name for p in region_factor_paths(TRAIN_REGIONS[0])]
    print("\n%-25s %15s %15s %15s" % ("factor", "mean", "std", "n_px(M)"))
    for n, m, s, c in zip(names, mean, std, cnt):
        print("%-25s %15.6f %15.6f %15.1f" % (n, m, s, c / 1e6))

    print("\n填入 config.py：")
    print("MEAN = [")
    for i in range(0, C, 5):
        print("    " + ", ".join("%.6f" % v for v in mean[i:i + 5]) + ",")
    print("]")
    print("STD = [")
    for i in range(0, C, 5):
        print("    " + ", ".join("%.6f" % v for v in std[i:i + 5]) + ",")
    print("]")


if __name__ == "__main__":
    main()
