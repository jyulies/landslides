# -*- coding: utf-8 -*-
"""
全图预测（ConvNeXt-Tiny, 13ch, 31x31 patch）
- 一次性载入 13 因子到内存 (C,H,W)
- 瓦片推理：每个瓦片切成 (输出网格 th×tw)，用 unfold 视图产生 patch，分块+分批送 GPU
- 预测输出为 class=1 的概率图（float32），GeoTIFF，LZW 压缩
- 默认不做 nodata 检查（你已预处理干净）。如需开启，把 CHECK_NODATA=True 并设置 NODATA_VAL
"""

import os, sys, math
from pathlib import Path
from typing import List, Tuple
import torch.nn.functional as F

import numpy as np
import torch
import rasterio
from rasterio import Affine
from tqdm.auto import tqdm

# Allow imports from the project root when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import (
    PROJECT_ROOT, FACTOR_PATHS, CHECKPOINTS_DIR, OUTPUT_DIR,
    MEAN, STD, PATCH_SIZE, STRIDE
)
from src.models.convnext import convnext_tiny_13ch_31

# =============== 配置 ===============
# 训练得到的权重（最新训练结果）
WEIGHTS = os.environ.get(
    "LANDSLIDE_WEIGHTS",
    str(CHECKPOINTS_DIR / "20260704-213937" / "best_convnext_tiny31_trans.pth")
)

# 输出概率图（class=1 概率）
OUT_TIF = os.environ.get(
    "LANDSLIDE_OUTPUT",
    str(OUTPUT_DIR / "prob_convnext31_trans.tif")
)

# Patch/Stride（和训练保持一致）
PATCH   = PATCH_SIZE

# 推理批大小（按显存调，P5000上 2048~4096 一般可行；OOM 就减小）
BATCH   = 2048

# 推理瓦片输出网格尺寸（th, tw），越大越快但更吃内存；建议 64~128
TILE_OUT_H = 128
TILE_OUT_W = 128

# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 归一化开关 & 统计量（用你新算的）
USE_NORM = True
MEAN = np.array(MEAN, dtype=np.float32)
STD  = np.array(STD,  dtype=np.float32)

# 是否检查 nodata（你已清理，这里默认关闭）
CHECK_NODATA = False
NODATA_VAL = np.nan   # 如需检查，改成实际 nodata 值

# =============== 模型定义 ===============
# 直接复用你的 convnext_tiny_13ch_31
import torch.nn as nn

# AMP 兼容导入
try:
    from torch.cuda.amp import autocast as amp_autocast
except Exception:
    from torch.amp import autocast as amp_autocast

# =============== 工具函数 ===============
def read_factors(paths: List[Path]) -> Tuple[np.ndarray, Affine, rasterio.crs.CRS]:
    """一次性读入所有因子到 (C,H,W)，假定都已对齐到同一网格"""
    bands = []
    transform = None
    crs = None
    print("📥 读取因子栅格…")
    with tqdm(total=len(paths), ncols=90) as bar:
        for fp in paths:
            with rasterio.open(fp) as src:
                if transform is None:
                    transform, crs = src.transform, src.crs
                if src.count == 1:
                    arr = src.read(1)
                    bands.append(arr)
                else:
                    arrs = src.read()  # (count, H, W)
                    bands.extend([a for a in arrs])
            bar.update(1)
    data = np.stack(bands, axis=0).astype(np.float32)  # (C,H,W)
    return data, transform, crs

def make_out_geo(transform: Affine, rows: int, cols: int, pad: int, stride: int) -> Affine:
    """输出概率图的仿射，像元中心与输入 patch 中心对齐"""
    return Affine(transform.a*stride, transform.b,
                  transform.c + pad*transform.a,
                  transform.d, transform.e*stride,
                  transform.f + pad*transform.e)

# =============== 主预测 ===============
@torch.no_grad()
def main():
    # 1) 数据
    data, transform, crs = read_factors(FACTOR_PATHS)  # (C,H,W)
    C, H, W = data.shape
    assert C == 13, f"需要13通道，当前为 C={C}"

    # 2) 输出网格尺寸（按中心滑窗，步长=STRIDE）
    pad = PATCH // 2
    out_rows = (H - PATCH) // STRIDE + 1
    out_cols = (W - PATCH) // STRIDE + 1
    print(f"✔ 因子 shape = ({C},{H},{W})  -> 输出概率图 = ({out_rows},{out_cols})")

    # 3) 模型
    net = convnext_tiny_13ch_31(num_classes=2, in_ch=13, pretrained=False).to(DEVICE).eval()
    state = torch.load(WEIGHTS, map_location=DEVICE)
    net.load_state_dict(state)
    net = net.to(memory_format=torch.channels_last)
    print("✔ 模型加载完毕")

    # 4) 输出缓存
    prob_map = np.empty((out_rows, out_cols), dtype=np.float32)

    # 5) 归一化 tensor
    if USE_NORM:
        mean_t = torch.from_numpy(MEAN).to(DEVICE).view(1, C, 1, 1)
        std_t  = torch.from_numpy(STD).to(DEVICE).view(1, C, 1, 1)

    # 6) 瓦片循环（按“输出网格”分块）
    tiles_r = math.ceil(out_rows / TILE_OUT_H)
    tiles_c = math.ceil(out_cols / TILE_OUT_W)
    total_tiles = tiles_r * tiles_c
    pbar = tqdm(total=total_tiles, desc="Tiles", ncols=90)

    for tr in range(0, out_rows, TILE_OUT_H):
        th = min(TILE_OUT_H, out_rows - tr)

        for tc in range(0, out_cols, TILE_OUT_W):
            tw = min(TILE_OUT_W, out_cols - tc)

            # ✅ 正确的输入切片边界（输出→输入，乘以 STRIDE）
            in_r0 = tr * STRIDE
            in_r1 = (tr + th - 1) * STRIDE + PATCH
            in_c0 = tc * STRIDE
            in_c1 = (tc + tw - 1) * STRIDE + PATCH

            tile_np = data[:, in_r0:in_r1, in_c0:in_c1]  # (C, Ht, Wt)
            x = torch.from_numpy(tile_np).to(DEVICE)  # (C, Ht, Wt)

            # unfold → (th, tw, C, P, P)
            ph = x.unfold(1, PATCH, STRIDE)  # (C, th, Wt, P)
            patches_view = ph.unfold(2, PATCH, STRIDE)  # (C, th, tw, P, P)
            patches_view = patches_view.permute(1, 2, 0, 3, 4).contiguous()

            # 保险：万一某块因为边界对齐问题没有 patch，就跳过
            if patches_view.numel() == 0:
                pbar.update(1)
                continue

            COL_STEP = 64
            tile_probs = np.empty((th, tw), dtype=np.float32)
            with amp_autocast(enabled=(DEVICE.type == 'cuda')):
                for c0 in range(0, tw, COL_STEP):
                    c1 = min(c0 + COL_STEP, tw)  # 本次处理的列宽 cw = c1-c0
                    # (th, cw, C, P, P) → (th*cw, C, P, P)
                    small = patches_view[:, c0:c1].contiguous().view(-1, C, PATCH, PATCH)

                    if USE_NORM:
                        small = (small - mean_t) / std_t
                    small = small.to(memory_format=torch.channels_last)

                    n_small = small.size(0)  # 应该等于 th * (c1-c0)
                    if n_small == 0:
                        # 防御：极端边界时跳过
                        tile_probs[:, c0:c1] = 0.5
                        continue

                    out_buf = []
                    for s in range(0, n_small, BATCH):
                        sb = small[s:s + BATCH]
                        y = net(sb)
                        prob = torch.softmax(y, dim=1)[:, 1]
                        out_buf.append(prob.detach().float().cpu().numpy())

                    # 回填到 (th, cw) 的正确位置
                    probs_chunk = np.concatenate(out_buf, axis=0)
                    assert probs_chunk.size == th * (
                                c1 - c0), f"size mismatch: got {probs_chunk.size}, expect {th * (c1 - c0)}"
                    tile_probs[:, c0:c1] = probs_chunk.reshape(th, c1 - c0)

            # 把本瓦片概率放回全图
            prob_map[tr:tr + th, tc:tc + tw] = tile_probs
            pbar.update(1)
    pbar.close()

    # ---------- 上采样回 stride=1（可选，建议 True） ----------
    UPSAMPLE_TO_STRIDE1 = True

    if UPSAMPLE_TO_STRIDE1 and STRIDE > 1:
        prob_t = torch.from_numpy(prob_map)[None, None]  # (1,1,h,w)
        prob_up = F.interpolate(
            prob_t,
            size=(H - PATCH + 1, W - PATCH + 1),
            mode='bilinear',
            align_corners=False
        )
        prob_map = prob_up[0, 0].cpu().numpy().astype('float32')

        # stride=1 的仿射（像元大小不乘 STRIDE）
        out_tr = Affine(
            transform.a, transform.b, transform.c + (PATCH // 2) * transform.a,
            transform.d, transform.e, transform.f + (PATCH // 2) * transform.e
        )
    else:
        # 保持低分辨率网格（像元大小 * STRIDE）
        out_tr = Affine(
            transform.a * STRIDE, transform.b,
            transform.c + (PATCH // 2) * transform.a,
            transform.d, transform.e * STRIDE,
            transform.f + (PATCH // 2) * transform.e
        )

    # ---------- 写 GeoTIFF ----------
    h, w = prob_map.shape  # 一律用实际数组尺寸，避免尺寸/仿射不匹配
    Path(OUT_TIF).parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": h,
        "width":  w,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": out_tr,
        "compress": "lzw",
        "nodata": None
    }
    with rasterio.open(OUT_TIF, "w", **profile) as dst:
        dst.write(prob_map, 1)

    print(f"🎉 完成！输出：{OUT_TIF} | shape={prob_map.shape} | STRIDE={STRIDE} | upsample={UPSAMPLE_TO_STRIDE1}")

# =============== 入口 ===============
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
