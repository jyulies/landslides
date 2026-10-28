# -*- coding: utf-8 -*-
"""
多区域样本采样：config.TRAIN_REGIONS 各区全部样本进训练集，
config.VAL_REGIONS 各区全部样本进验证集（不做区内的 8:2 划分）。

复用 extract_patches.py 的采样函数；与单区版的差异：
  - shp 坐标系与因子栅格不一致时，先用 transform_geom 重投影写临时 shp
    （如 wanzhou 的 EPSG:3857 → 32649、nanjiang 的 EPSG:4326 → 32648）
  - 输出到 config.SAMPLE_MULTI_DIR/<region>/{train,val}_points.json，
    格式与 extract_patches.py 的 *_points.json 一致，RasterPatchDataset 直读

用法：
    python src/data/extract_patches_multi.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import shapefile  # pyshp
from rasterio.warp import transform_geom
from tqdm import tqdm

# Allow imports from the project root when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import (
    MODEL_DIR, SAMPLE_MULTI_DIR, SEED, PATCH_SIZE, NONLS_BUFFER_DIST,
    N_NONLS_RATIO, TRAIN_REGIONS, VAL_REGIONS, region_factor_paths,
)
from src.data.extract_patches import (
    generate_ls_points_from_shp, build_ls_mask, generate_nonls_points,
    extract_patch, coord_to_rowcol,
)

PAD = PATCH_SIZE // 2


def find_shp(region):
    """model/<region>/shp/ 下唯一的滑坡面 shp"""
    shps = sorted((MODEL_DIR / region / "shp").glob("*.shp"))
    assert shps, "未找到 %s 的滑坡 shp" % region
    return shps[0]


def ensure_crs_matched_shp(shp_path, dst_crs, work_dir):
    """shp 与栅格 CRS 不一致时重投影到 work_dir 下的临时 shp，返回可用路径"""
    prj = shp_path.with_suffix(".prj")
    src_crs = None
    if prj.exists():
        try:
            src_crs = rasterio.crs.CRS.from_wkt(prj.read_text(errors="ignore"))
        except Exception:
            pass
    if src_crs is None or src_crs == dst_crs:
        return shp_path

    out = work_dir / (shp_path.stem + "_reproj.shp")
    sf = shapefile.Reader(str(shp_path))
    # 统一写 2D Polygon（源可能是 PolygonZ/M 等，下游只需平面几何）
    w = shapefile.Writer(str(out), shapeType=shapefile.POLYGON)
    w.fields = sf.fields[1:]  # 跳过 DeletionFlag
    for shp, rec in zip(sf.shapes(), sf.records()):
        g = transform_geom(src_crs, dst_crs, shp.__geo_interface__)
        # Polygon: coordinates = [ring, ...]；MultiPolygon: [[ring, ...], ...]
        # shp 格式不区分两者——多部件面就是多 part，全部环作为 parts 写入即可
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        rings = [ring for poly in polys for ring in poly]
        w.poly([[list(pt[:2]) for pt in ring] for ring in rings])
        w.record(*rec)
    w.close()
    print("  重投影 shp: %s (%s -> %s)" % (shp_path.name, src_crs.to_epsg(), dst_crs.to_epsg()))
    return out


def sample_region(region, role):
    """role: 'train' | 'val'；返回写入的 points json 路径"""
    factor_paths = region_factor_paths(region)
    for fp in factor_paths:
        assert fp.is_file(), "缺因子: %s" % fp

    bands = []
    for fp in factor_paths:
        with rasterio.open(fp) as src:
            bands.append(src.read(1).astype(np.float32))
            transform, crs = src.transform, src.crs
    data = np.stack(bands, axis=0)
    C, H, W = data.shape
    print("\n=== %s [%s] 因子 (%d,%d,%d) crs=%s" % (region, role, C, H, W, crs.to_epsg()))

    out_dir = SAMPLE_MULTI_DIR / region
    out_dir.mkdir(parents=True, exist_ok=True)
    shp_path = ensure_crs_matched_shp(find_shp(region), crs, out_dir)

    # 正样本：面内多点；负样本：避面随机
    ls_points = generate_ls_points_from_shp(shp_path, transform, H, W)
    n_target = int(len(ls_points) * N_NONLS_RATIO)
    ls_mask = build_ls_mask(shp_path, transform, H, W, buffer_dist=NONLS_BUFFER_DIST)
    nonls_coords = generate_nonls_points(transform, H, W, ls_mask, n_target, seed=SEED)

    # 切 patch 验 NaN（与原逻辑一致：含 NaN 丢弃；边界点已在采样时排除/在此复查）
    records = []
    n_nan = 0
    for i, (pid, x, y, j) in enumerate(tqdm(ls_points, desc="  LS patches")):
        r, c = coord_to_rowcol(transform, x, y)
        if not (PAD <= r < H - PAD and PAD <= c < W - PAD):
            continue
        if not np.isfinite(extract_patch(data, r, c)).all():
            n_nan += 1
            continue
        records.append({"name": "ls_%05d_%d" % (pid, i), "x": x, "y": y,
                        "label": 1, "pid": pid, "j": j})
    for i, (x, y) in enumerate(tqdm(nonls_coords, desc="  non-LS patches")):
        r, c = coord_to_rowcol(transform, x, y)
        if not (PAD <= r < H - PAD and PAD <= c < W - PAD):
            continue
        if not np.isfinite(extract_patch(data, r, c)).all():
            n_nan += 1
            continue
        records.append({"name": "nols_%05d" % i, "x": x, "y": y,
                        "label": 0, "pid": -1, "j": 0})

    n_pos = sum(r["label"] for r in records)
    out_json = out_dir / ("%s_points.json" % role)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f)
    print("  -> %s: 正 %d + 负 %d = %d（NaN 丢弃 %d）"
          % (out_json.name, n_pos, len(records) - n_pos, len(records), n_nan))
    return out_json


def main():
    assert TRAIN_REGIONS and VAL_REGIONS, "config.TRAIN_REGIONS/VAL_REGIONS 未配置"
    for r in TRAIN_REGIONS:
        sample_region(r, "train")
    for r in VAL_REGIONS:
        sample_region(r, "val")
    print("\n完成，输出目录: %s" % SAMPLE_MULTI_DIR)


if __name__ == "__main__":
    main()
