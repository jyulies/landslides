# -*- coding: utf-8 -*-
"""
从永川滑坡多边形 shapefile 提取质心，生成与武平流程兼容的点 CSV。

输出列：x, y, area_m2, perimeter_m（x/y 为投影坐标 EPSG:32648，
area/perimeter 来自 shapefile 属性表，便于后续分析筛选）。

用法：
    python src/data/shp_centroids_to_csv.py
"""

import csv
import sys
from pathlib import Path

import shapefile  # pyshp

# Allow imports from the project root when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import PROJECT_ROOT

SHP_PATH = PROJECT_ROOT / "Dataset" / "yonchuan" / "shp" / "landslides.shp"
OUT_CSV = PROJECT_ROOT / "Dataset" / "yonchuan" / "yongchuan_landslide_centroids.csv"


def polygon_centroid(points):
    """有符号面积加权的环质心（points 为闭合或不闭合的 (x,y) 序列）。"""
    a = cx = cy = 0.0
    n = len(points)
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        a += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    a *= 0.5
    if abs(a) < 1e-12:  # 退化多边形，退回顶点均值
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return sum(xs) / n, sum(ys) / n
    return cx / (6 * a), cy / (6 * a)


def shape_centroid(shape):
    """多部分多边形：按各环 |面积| 加权合并质心（外环/内环用符号面积）。"""
    parts = list(shape.parts) + [len(shape.points)]
    total_a = cx_sum = cy_sum = 0.0
    for i in range(len(parts) - 1):
        ring = shape.points[parts[i]:parts[i + 1]]
        if len(ring) < 3:
            continue
        # 复用 polygon_centroid，但需要面积；这里重新算一次
        a = 0.0
        for j in range(len(ring)):
            x0, y0 = ring[j]
            x1, y1 = ring[(j + 1) % len(ring)]
            a += x0 * y1 - x1 * y0
        a *= 0.5
        cx, cy = polygon_centroid(ring)
        total_a += a
        cx_sum += cx * a
        cy_sum += cy * a
    if abs(total_a) < 1e-12:
        xs = [p[0] for p in shape.points]
        ys = [p[1] for p in shape.points]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    return cx_sum / total_a, cy_sum / total_a


def main():
    sf = shapefile.Reader(str(SHP_PATH))
    field_names = [f[0] for f in sf.fields[1:]]
    i_area = field_names.index("Shape_Area") if "Shape_Area" in field_names else None
    i_len = field_names.index("Shape_Leng") if "Shape_Leng" in field_names else None

    rows = []
    for sr in sf.iterShapeRecords():
        cx, cy = shape_centroid(sr.shape)
        area = sr.record[i_area] if i_area is not None else ""
        perim = sr.record[i_len] if i_len is not None else ""
        rows.append((cx, cy, area, perim))

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "area_m2", "perimeter_m"])
        w.writerows(rows)

    print("共 %d 个滑坡多边形 -> %s" % (len(rows), OUT_CSV))


if __name__ == "__main__":
    main()
