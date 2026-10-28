# -*- coding: utf-8 -*-
"""
跨区域迁移评估：用永川训练的模型在新区出的概率图 + 新区滑坡编目 shp，
定量计算迁移效果（ROC AUC / PR AUC）。

正样本：滑坡面覆盖的概率图像元（all_touched=True，宁多勿漏）
负样本：掩膜外（可选外扩缓冲）随机采样等数量像元
NaN 像元（概率图边界）在两类中都排除

用法：
    python src/inference/eval_transfer.py \
        --prob output/prob_<新区>.tif \
        --shp  Dataset/<新区>/shp/landslides.shp

注意：shp 可以是任意坐标系——读取 .prj 后自动重投影到概率图 CRS
（无 .prj 时假定与概率图一致，并做重叠度 sanity check）。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
import shapefile  # pyshp
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from scipy import ndimage
from sklearn.metrics import roc_auc_score, average_precision_score

# Allow imports from the project root when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    ap = argparse.ArgumentParser(description="迁移评估：概率图 vs 新区滑坡编目")
    ap.add_argument("--prob", required=True, help="predict.py 输出的概率图 tif")
    ap.add_argument("--shp", required=True, help="新区滑坡面 shapefile")
    ap.add_argument("--buffer", type=int, default=2,
                    help="负样本与滑坡面的缓冲距离（概率图像元数，默认 2）")
    ap.add_argument("--neg-ratio", type=float, default=1.0, help="负样本数 = 正样本数 × 该比例")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 1) 读概率图
    with rasterio.open(args.prob) as src:
        prob = src.read(1).astype(np.float32)
        transform = src.transform
        bounds = src.bounds
        crs = src.crs
    H, W = prob.shape
    print("概率图: %dx%d, crs=%s" % (W, H, crs))

    # 2) 滑坡面栅格化到概率图网格（shp 任意 CRS，自动重投影）
    sf = shapefile.Reader(str(args.shp))
    prj = Path(str(args.shp).replace(".shp", ".prj"))
    src_crs = None
    if prj.exists():
        try:
            src_crs = rasterio.crs.CRS.from_wkt(prj.read_text(errors="ignore"))
        except Exception:
            print("!! 无法解析 %s，假定 shp 与概率图同坐标系" % prj.name)
    else:
        print("!! 缺少 .prj，假定 shp 与概率图同坐标系")
    if src_crs and src_crs != crs:
        print("shp 坐标系 %s != 概率图 %s，自动重投影" % (src_crs.to_epsg() or src_crs, crs.to_epsg() or crs))
    geoms = []
    for shp in sf.shapes():
        g = shp.__geo_interface__
        if src_crs and src_crs != crs:
            g = transform_geom(src_crs, crs, g)
        geoms.append(g)
    mask = rasterize(geoms, out_shape=(H, W), transform=transform, fill=0,
                     default_value=1, all_touched=True, dtype="uint8").astype(bool)
    n_pos_px = int(mask.sum())
    print("滑坡面像元: %d (%.2f%%)" % (n_pos_px, 100.0 * n_pos_px / (H * W)))
    if n_pos_px == 0:
        print("!! 滑坡面与概率图零重叠——坐标系很可能不一致，请检查 shp 的 CRS")
        sys.exit(1)

    # 3) 正负样本（排除 NaN；负样本外扩 buffer）
    valid = np.isfinite(prob)
    neg_pool = valid & ~ndimage.binary_dilation(mask, iterations=args.buffer)
    pos_idx = np.argwhere(mask & valid)
    rng = np.random.RandomState(args.seed)
    n_neg = min(int(len(pos_idx) * args.neg_ratio), int(neg_pool.sum()))
    neg_idx_all = np.argwhere(neg_pool)
    neg_idx = neg_idx_all[rng.choice(len(neg_idx_all), size=n_neg, replace=False)]

    y_true = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))])
    y_score = np.concatenate([
        prob[pos_idx[:, 0], pos_idx[:, 1]],
        prob[neg_idx[:, 0], neg_idx[:, 1]],
    ])

    # 4) 指标
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    pos_p = prob[pos_idx[:, 0], pos_idx[:, 1]]
    print("\n=== 迁移评估结果 ===")
    print("正样本像元: %d | 负样本像元: %d" % (len(pos_idx), len(neg_idx)))
    print("ROC AUC = %.4f | PR AUC(AP) = %.4f" % (auc, ap))
    print("滑坡面内概率分位数: p10=%.3f p50=%.3f p90=%.3f" % tuple(
        np.percentile(pos_p, [10, 50, 90])))
    print("\n参考解读：AUC>0.8 迁移良好；0.7~0.8 可接受；<0.7 建议用新区样本微调")


if __name__ == "__main__":
    main()
