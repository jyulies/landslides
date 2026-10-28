# -*- coding: utf-8 -*-
"""
用研究区边界 shp 裁剪已对齐的因子栅格：边界外像元置为 NaN。

shp 可以是任意坐标系（自动重投影到栅格网格 CRS，如 WGS84 经纬度的
area.shp 可直接用）；网格本身不变（行列数、分辨率、范围保持），
只是把研究区外置为 nodata，与全流程的 NaN 约定一致。

用法：
    python src/data/clip_by_area.py \
        --dir Dataset/<区域>/factor_aligned \
        --shp Dataset/<区域>/area/area.shp
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
import shapefile  # pyshp
from rasterio.features import rasterize
from rasterio.warp import transform_geom

# Allow imports from the project root when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_area_geom(shp_path, dst_crs):
    """读 shp 全部面要素，重投影到 dst_crs，返回几何列表"""
    sf = shapefile.Reader(str(shp_path))
    src_crs = None
    prj = Path(str(shp_path).replace(".shp", ".prj"))
    if prj.exists():
        src_crs = rasterio.crs.CRS.from_wkt(prj.read_text(errors="ignore"))
    geoms = []
    for shp in sf.shapes():
        g = shp.__geo_interface__
        if src_crs and src_crs != dst_crs:
            g = transform_geom(src_crs, dst_crs, g)
        geoms.append(g)
    return geoms


def main():
    ap = argparse.ArgumentParser(description="按研究区边界裁剪因子栅格（界外置 NaN）")
    ap.add_argument("--dir", required=True, help="已对齐因子目录（原地裁剪）")
    ap.add_argument("--shp", required=True, help="研究区边界 shp（任意 CRS）")
    args = ap.parse_args()

    tifs = sorted(Path(args.dir).glob("*.tif"))
    if not tifs:
        print("!! 目录下没有 tif: %s" % args.dir)
        sys.exit(1)

    # 以第一个栅格的网格为准栅格化边界
    with rasterio.open(tifs[0]) as s:
        H, W, transform, crs = s.height, s.width, s.transform, s.crs
    geoms = load_area_geom(args.shp, crs)
    mask = rasterize(geoms, out_shape=(H, W), transform=transform, fill=0,
                     default_value=1, all_touched=True, dtype="uint8").astype(bool)
    print("边界像元: %d/%d (%.1f%%)  crs=%s" % (mask.sum(), H * W, 100 * mask.mean(), crs))
    if mask.sum() == 0:
        print("!! 边界与栅格零重叠——请检查 shp 坐标系")
        sys.exit(1)

    for f in tifs:
        with rasterio.open(f) as s:
            arr = s.read(1)
            profile = s.profile.copy()
            if (s.height, s.width) != (H, W):
                print("  !! 跳过 %s：网格不一致 %dx%d" % (f.name, s.width, s.height))
                continue
        before = np.isfinite(arr).sum()
        arr[~mask] = np.nan
        after = np.isfinite(arr).sum()
        profile.update(nodata=np.nan)
        with rasterio.open(f, "w", **profile) as d:
            d.write(arr, 1)
        print("  %-25s 有效 %d -> %d (保留 %.1f%%)" % (
            f.name, before, after, 100 * after / max(before, 1)))

    print("\n完成，已原地裁剪: %s" % args.dir)


if __name__ == "__main__":
    main()
