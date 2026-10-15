# -*- coding: utf-8 -*-
"""
LandslideDataset - 加载训练/验证样本（.npy 文件模式）

RasterPatchDataset - 内存直读模式：因子栅格一次性读入内存，按需切 patch，
                     无需 .npy 文件（适合大内存服务器）。

JSON格式（示例）:
{
    "sample_00001": 1,
    "sample_00002": 0,
    ...
}

数据文件: ROOT/sample_00001.npy  -> shape=(C,31,31), dtype=float32

采样点 JSON（RasterPatchDataset 使用，extract_patches.py 生成）:
[{"name": "ls_00001", "x": 586945.0, "y": 3251958.6, "label": 1}, ...]

Reference / 引用：
    Luo, S., Mao, W., Yang, Z., Zheng, G., He, Z., Wang, J., & Huang, Y. (2026).
    CNXT-Ti-LT--Based Multi-Scale Feature--Aware Susceptibility Mapping of
    Rainfall-Induced Clustered Landslides in Southeast China.
    Journal of Geophysical Research: Machine Learning and Computation,
    3, e2025JH001115. https://doi.org/10.1029/2025JH001115

    Training dataset / 训练样本与因子: https://doi.org/10.5281/zenodo.18463063
"""
import os
import json
import numpy as np
import rasterio
import torch
from rasterio.transform import rowcol
from torch.utils.data import Dataset


class LandslideDataset(Dataset):
    def __init__(self, root, json_path, transform=None):
        """
        Args:
            root: 样本npy文件所在目录
            json_path: 标签JSON文件路径
            transform: 可选的数据变换
        """
        self.root = root
        with open(json_path, "r", encoding="utf-8") as f:
            self.samples = list(json.load(f).items())
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, label = self.samples[idx]
        path = os.path.join(self.root, name + ".npy")
        img = np.load(path).astype(np.float32)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

    @staticmethod
    def collate_fn(batch):
        imgs, labels = zip(*batch)
        return torch.stack(imgs, 0), torch.stack(labels, 0)


class RasterPatchDataset(Dataset):
    """
    内存直读模式：所有因子栅格一次读入内存 (C,H,W)，__getitem__ 时按
    采样点坐标现切 (C,PATCH,PATCH) patch（边界零填充，与 extract_patch 一致）。

    永川 16 通道数据约 31MB，内存开销可忽略；Linux 下 DataLoader 多进程
    通过 fork 共享内存（写时复制），不会按 worker 复制。
    """

    def __init__(self, factor_paths, points_json, transform=None, patch=31):
        """
        Args:
            factor_paths: 因子 TIF 路径列表（顺序即通道顺序）
            points_json: 采样点 JSON 路径（extract_patches.py 生成的 *_points.json）
            transform: 可选的数据变换
            patch: patch 边长（默认 31）
        """
        bands = []
        geo_transform = None
        for fp in factor_paths:
            with rasterio.open(fp) as src:
                bands.append(src.read(1).astype(np.float32))
                if geo_transform is None:
                    geo_transform = src.transform
        self.data = np.stack(bands, axis=0)  # (C, H, W)
        self.geo_transform = geo_transform
        self.patch = patch
        self.pad = patch // 2
        self.transform = transform

        with open(points_json, "r", encoding="utf-8") as f:
            pts = json.load(f)
        # 预计算行列号，__getitem__ 无需再做坐标转换
        self.items = []
        for p in pts:
            r, c = rowcol(geo_transform, p["x"], p["y"])
            self.items.append((int(r), int(c), int(p["label"])))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        row, col, label = self.items[idx]
        C, H, W = self.data.shape
        pad = self.pad
        r0, r1 = row - pad, row + pad + 1
        c0, c1 = col - pad, col + pad + 1

        patch = np.zeros((C, self.patch, self.patch), dtype=np.float32)
        pr0, pr1 = max(0, r0), min(H, r1)
        pc0, pc1 = max(0, c0), min(W, c1)
        ppr0 = pr0 - r0
        ppr1 = ppr0 + (pr1 - pr0)
        ppc0 = pc0 - c0
        ppc1 = ppc0 + (pc1 - pc0)
        patch[:, ppr0:ppr1, ppc0:ppc1] = self.data[:, pr0:pr1, pc0:pc1]

        if self.transform:
            patch = self.transform(patch)
        return patch, torch.tensor(label, dtype=torch.long)

    @staticmethod
    def collate_fn(batch):
        imgs, labels = zip(*batch)
        return torch.stack(imgs, 0), torch.stack(labels, 0)
