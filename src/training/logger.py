# -*- coding: utf-8 -*-
"""
简化版论文logger - 保留接口兼容，不依赖原实现

Reference / 引用：
    Luo, S., Mao, W., Yang, Z., Zheng, G., He, Z., Wang, J., & Huang, Y. (2026).
    CNXT-Ti-LT--Based Multi-Scale Feature--Aware Susceptibility Mapping of
    Rainfall-Induced Clustered Landslides in Southeast China.
    Journal of Geophysical Research: Machine Learning and Computation,
    3, e2025JH001115. https://doi.org/10.1029/2025JH001115
"""
import json
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class RunConfig:
    seed: int
    model: str
    in_ch: int
    patch: int
    epochs: int
    batch: int
    lr: float
    weight_decay: float
    optimizer: str
    scheduler: str
    dataset_root: str
    train_json: str
    val_json: str
    mean: list
    std: list
    augment: str
    device: str
    torch: str = ""
    cuda: str = ""
    cudnn: int = -1
    gpu_name: str = ""


class PaperLogger:
    def __init__(self, paper_dir, cfg: RunConfig):
        self.paper_dir = Path(paper_dir)
        self.paper_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        with open(self.paper_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
        self.val_preds = []
        self.val_labels = []

    def add_val_batch(self, outputs, labels):
        """收集验证批次结果（用于论文级统计）"""
        probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
        self.val_preds.extend(probs)
        self.val_labels.extend(labels.cpu().numpy())

    def finalize_epoch(self, epoch, lr, tr_loss, tr_acc, vl_loss, vl_acc,
                       epoch_sec=0, max_mem=0, train_samples=0):
        """每轮结束调用，打印日志"""
        self.val_preds = []
        self.val_labels = []
