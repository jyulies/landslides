# -*- coding: utf-8 -*-
"""
深度诊断：检查patch内容、坐标匹配、TIF属性

Reference / 引用：
    Luo, S., Mao, W., Yang, Z., Zheng, G., He, Z., Wang, J., & Huang, Y. (2026).
    CNXT-Ti-LT--Based Multi-Scale Feature--Aware Susceptibility Mapping of
    Rainfall-Induced Clustered Landslides in Southeast China.
    Journal of Geophysical Research: Machine Learning and Computation,
    3, e2025JH001115. https://doi.org/10.1029/2025JH001115
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import rasterio

# Allow imports from the project root when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import SAMPLE_DIR, FACTOR_PATHS

SAMPLE_DIR = str(SAMPLE_DIR)

def main():
    print("=" * 60)
    print("深度诊断")
    print("=" * 60)

    # 1. TIF基本信息
    print("\n1. TIF坐标系与栅格信息")
    for i, fp in enumerate(FACTOR_PATHS):
        with rasterio.open(fp) as src:
            print(f"   {i+1}. {os.path.basename(fp):<25} CRS={src.crs}  Shape={src.shape}  Dtype={src.dtypes[0]}")

    # 2. 读取一个样本patch详细检查
    print("\n2. 样本patch详细检查")
    with open(os.path.join(SAMPLE_DIR, "train_labels.json")) as f:
        labels = json.load(f)
    names = list(labels.keys())

    # 取前5个样本
    for name in names[:5]:
        patch = np.load(os.path.join(SAMPLE_DIR, name + ".npy"))
        label = labels[name]
        zero_ratio = (patch == 0).sum() / patch.size
        nan_ratio = np.isnan(patch).sum() / patch.size
        print(f"\n   {name} (label={label}) shape={patch.shape}")
        print(f"   零值比例: {zero_ratio:.2%}  NaN比例: {nan_ratio:.2%}")
        print(f"   每通道零值比例:")
        factor_names = ['Elev','Slope','Aspect','TPI','Land','TRI','ProfC','PlanC','TWI','Litho','DTF','NDVI','DTR']
        for i, fn in enumerate(factor_names):
            zr = (patch[i] == 0).sum() / (31*31)
            print(f"     {fn}: {zr:.1%}  range=[{patch[i].min():.2f}, {patch[i].max():.2f}]")

    # 3. 全样本零值统计
    print("\n3. 全样本零值统计 (前500个)")
    zero_ratios = []
    for name in names[:500]:
        patch = np.load(os.path.join(SAMPLE_DIR, name + ".npy"))
        zero_ratios.append((patch == 0).sum() / patch.size)
    zero_ratios = np.array(zero_ratios)
    print(f"   平均零值比例: {zero_ratios.mean():.2%}")
    print(f"   中位数: {np.median(zero_ratios):.2%}")
    print(f"   >50%零值的样本: {(zero_ratios > 0.5).sum()}/500")
    print(f"   >80%零值的样本: {(zero_ratios > 0.8).sum()}/500")
    print(f"   100%零值的样本: {(zero_ratios == 1.0).sum()}/500")

    # 4. 检查标签与patch内容的关系
    print("\n4. 标签-内容关联检查")
    ls_patches = []
    nonls_patches = []
    for name in names[:200]:
        patch = np.load(os.path.join(SAMPLE_DIR, name + ".npy"))
        if labels[name] == 1:
            ls_patches.append(patch)
        else:
            nonls_patches.append(patch)
    
    ls_mean = np.mean([p.mean() for p in ls_patches])
    nonls_mean = np.mean([p.mean() for p in nonls_patches])
    print(f"   滑坡样本平均亮度: {ls_mean:.2f}")
    print(f"   非滑坡样本平均亮度: {nonls_mean:.2f}")
    print(f"   差异: {abs(ls_mean - nonls_mean):.2f}")
    if abs(ls_mean - nonls_mean) < 1:
        print("   ! 两类样本亮度几乎相同 - 标签与数据可能不匹配")

    # 5. 检查相邻样本是否相同（坐标转换可能重复）
    print("\n5. 样本重复性检查 (前100个)")
    dup_count = 0
    for i in range(min(100, len(names)-1)):
        p1 = np.load(os.path.join(SAMPLE_DIR, names[i] + ".npy"))
        p2 = np.load(os.path.join(SAMPLE_DIR, names[i+1] + ".npy"))
        if np.allclose(p1, p2):
            dup_count += 1
    print(f"   相邻重复样本: {dup_count}/99")
    if dup_count > 10:
        print("   ! 大量重复 - 坐标转换可能有问题")

if __name__ == "__main__":
    main()
