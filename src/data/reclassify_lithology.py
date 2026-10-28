# -*- coding: utf-8 -*-
"""
重写岩性栅格：用地质图 shp 的 DSN（地层名称）字段，把各区岩性重分类到
永川训练区的 9 类图例，直接栅格化到该区已对齐网格并覆盖 Lithology.tif。

永川图例（由 shp 叠加原栅格反推得到）：
  0 = 其他/未覆盖（第四系、水面、图幅外）
  1 = 下沙溪庙组   2 = 新田沟组     3 = 须家河组   4 = 珍珠冲组
  5 = 自流井组     6 = 上沙溪庙组下段  7 = 上沙溪庙组上段  8 = 嘉陵江组

匹配规则：名称规范化（三迭→三叠、二迭→二叠、中—下→中下、去空格）后包含匹配；
未分段的上沙溪庙组默认归 6（下段，永川主体）；识别不了的一律 0。
分类因子，最近邻语义：直接按面栅格化，无插值。

用法：
    python src/data/reclassify_lithology.py           # model/ 下五区全做
"""
import re
import sys
from pathlib import Path

import numpy as np
import rasterio
import shapefile  # pyshp
from rasterio.features import rasterize
from rasterio.warp import transform_geom

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MODEL_DIR = Path("Dataset/model")

# (匹配关键词, 类别)——按顺序优先匹配，关键词取规范化后的地层名子串
RULES = [
    ("上沙溪庙组上段", 7), ("沙溪庙组上段", 7),
    ("上沙溪庙组下段", 6), ("沙溪庙组下段", 6),
    ("上沙溪庙组",     6),   # 未分段 → 下段（永川主体）
    ("下沙溪庙组",     1),
    ("新田沟组",       2),
    ("须家河组",       3),
    ("珍珠冲组",       4),
    ("自流井组",       5),
    ("嘉陵江组",       8),
]


def normalize(name):
    if not name:
        return ""
    s = name.replace("三迭系", "三叠系").replace("二迭系", "二叠系")
    s = s.replace("中—下统", "中下统").replace(" ", "").strip()
    return s


def classify(dsn):
    s = normalize(dsn)
    for kw, cls in RULES:
        if kw in s:
            return cls
    return 0


def process_region(region_dir):
    lith_dir = region_dir / "Lithology"
    shps = list(lith_dir.glob("*.shp")) if lith_dir.is_dir() else []
    ref = region_dir / "factor_aligned" / "Lithology.tif"
    if not shps or not ref.is_file():
        print("!! %s: 缺 Lithology shp 或已对齐栅格，跳过" % region_dir.name)
        return
    shp_path = shps[0]

    with rasterio.open(ref) as s:
        old = s.read(1)
        H, W, tf, crs, profile = s.height, s.width, s.transform, s.crs, s.profile.copy()

    sf = shapefile.Reader(str(shp_path))
    fields = [f[0] for f in sf.fields[1:]]
    i_dsn = fields.index("DSN")
    src_crs = None
    prj = shp_path.with_suffix(".prj")
    if prj.exists():
        src_crs = rasterio.crs.CRS.from_wkt(prj.read_text(errors="ignore"))

    geoms, unmapped = [], {}
    for shp, rec in zip(sf.shapes(), sf.records()):
        cls = classify(rec[i_dsn])
        if cls == 0:
            unmapped[rec[i_dsn] or "(空)"] = unmapped.get(rec[i_dsn] or "(空)", 0) + 1
        g = shp.__geo_interface__
        if src_crs and src_crs != crs:
            g = transform_geom(src_crs, crs, g)
        geoms.append((g, cls))

    new = rasterize(geoms, out_shape=(H, W), transform=tf, fill=0,
                    dtype="float32")
    # 研究区外（其他因子的 NaN 区）保持 NaN，与流水线约定一致
    new[~np.isfinite(old)] = np.nan

    profile.update(dtype="float32", nodata=np.nan, compress="lzw")
    with rasterio.open(ref, "w", **profile) as d:
        d.write(new.astype(np.float32), 1)

    valid = np.isfinite(new)
    vals, cnt = np.unique(new[valid], return_counts=True)
    dist = ", ".join("%d:%.1f%%" % (v, 100 * c / valid.sum()) for v, c in zip(vals, cnt))
    print("✔ %-12s 类别分布: %s" % (region_dir.name, dist))
    if unmapped:
        top = sorted(unmapped.items(), key=lambda x: -x[1])[:8]
        print("    归为 0 的地层: %s%s" % (
            "、".join("%s(%d面)" % u for u in top),
            " 等" if len(unmapped) > 8 else ""))


def main():
    for d in sorted(MODEL_DIR.iterdir()):
        if d.is_dir():
            process_region(d)


if __name__ == "__main__":
    main()
