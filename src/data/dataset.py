# -*- coding: utf-8 -*-
"""
LandslideDataset - 加载训练/验证样本

JSON格式（示例）:
{
    "sample_00001": 1,
    "sample_00002": 0,
    ...
}

数据文件: ROOT/sample_00001.npy  -> shape=(13,31,31), dtype=float32
"""
import os
import json
import numpy as np
import torch
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
