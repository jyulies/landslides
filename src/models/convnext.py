# -*- coding: utf-8 -*-
"""
ConvNeXt-Tiny (13-channel, binary) for small patches (e.g., 31x31)
- Inflate first conv from 3->13 with RGB weights preserved
- Small-stride variant: stem stride 4 -> 2, last downsample stride 2 -> 1
  => total downsample = 8 (31/8 -> 3), keeps spatial info for small patches
- Added: dropout argument in classifier head

Reference / 引用：
    Luo, S., Mao, W., Yang, Z., Zheng, G., He, Z., Wang, J., & Huang, Y. (2026).
    CNXT-Ti-LT--Based Multi-Scale Feature--Aware Susceptibility Mapping of
    Rainfall-Induced Clustered Landslides in Southeast China.
    Journal of Geophysical Research: Machine Learning and Computation,
    3, e2025JH001115. https://doi.org/10.1029/2025JH001115

    Official model code / 官方代码: https://doi.org/10.5281/zenodo.17509051
"""

from typing import Optional
import torch
import torch.nn as nn
import torchvision.models as tv


class LayerNorm2d(nn.Module):
    """对 (N,C,H,W) 在通道维做 LayerNorm。"""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class DropPath(nn.Module):
    """Per-sample drop-path。"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x.div(keep) * mask

class LiteTransformerBlock(nn.Module):
    """
    轻量 Transformer：
    - 先用深度可分离 3×3 卷积做轻量位置编码
    - 再做一次全局 MHA（tokens = H*W，embed 先用通道做线性投影到更小维度）
    - 末尾一个 1×1 的 MLP（Conv 版）
    """
    def __init__(
        self,
        in_ch: int,
        attn_dim: int = 192,     # 显著小于 in_ch
        num_heads: int = 3,      # 保证 attn_dim % heads == 0
        mlp_ratio: float = 1.5,  # 很小，控制参数量
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        assert attn_dim % num_heads == 0, "attn_dim 应能被 num_heads 整除"

        # 位置编码（DWConv）
        self.pos = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch)
        # 注意力前归一化
        self.norm1 = LayerNorm2d(in_ch)

        # 通道 → 注意力嵌入维度
        self.q = nn.Linear(in_ch, attn_dim)
        self.k = nn.Linear(in_ch, attn_dim)
        self.v = nn.Linear(in_ch, attn_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim, num_heads=num_heads,
            dropout=attn_dropout, batch_first=True
        )
        self.proj = nn.Linear(attn_dim, in_ch)
        self.proj_drop = nn.Dropout(proj_dropout)
        self.drop_path = DropPath(drop_path)

        # MLP（1×1 conv）
        hidden = int(in_ch * mlp_ratio)
        self.norm2 = LayerNorm2d(in_ch)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, in_ch, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 轻量位置编码
        x = x + self.pos(x)

        # MHA
        y = self.norm1(x)
        y = y.permute(0, 2, 3, 1).reshape(B, H * W, C)   # (B, L, C), L=H*W
        q, k, v = self.q(y), self.k(y), self.v(y)
        z, _ = self.attn(q, k, v, need_weights=False)    # (B, L, attn_dim)
        z = self.proj_drop(self.proj(z))
        z = z.view(B, H, W, C).permute(0, 3, 1, 2)
        x = x + self.drop_path(z)                        # 残差1

        # MLP
        z2 = self.mlp(self.norm2(x))
        x = x + self.drop_path(z2)                       # 残差2
        return x

def _get_cnext_tiny_weights(pretrained: bool):
    if not pretrained:
        return None
    # 兼容不同 torchvision 版本
    if hasattr(tv, "ConvNeXt_Tiny_Weights"):
        return tv.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    return None  # 老版本兜底（部分旧版不支持权重枚举）


def _inflate_first_conv(conv_rgb: nn.Conv2d, in_ch: int) -> nn.Conv2d:
    """把 3 通道卷积权重扩展到 in_ch：前3个拷贝原权重，其余用RGB均值填充。"""
    new = nn.Conv2d(
        in_channels=in_ch,
        out_channels=conv_rgb.out_channels,
        kernel_size=conv_rgb.kernel_size,
        stride=conv_rgb.stride,
        padding=conv_rgb.padding,
        bias=(conv_rgb.bias is not None),
    )
    with torch.no_grad():
        # conv_rgb.weight: (out, 3, k, k)
        w = conv_rgb.weight
        # 用 RGB 的均值初始化额外通道
        mean_w = w.mean(dim=1, keepdim=True)          # (out,1,k,k)
        new_w = mean_w.repeat(1, in_ch, 1, 1)         # (out,in_ch,k,k)
        # 前 3 个通道直接拷贝
        c = min(3, in_ch)
        new_w[:, :c] = w[:, :c]
        new.weight.copy_(new_w)
        if conv_rgb.bias is not None:
            new.bias.copy_(conv_rgb.bias)
    return new


def convnext_tiny_13ch_31(
    num_classes: int = 2,
    in_ch: int = 13,
    pretrained: bool = True,
    *,
    small_stride: bool = True,
    drop_path_rate: float = 0.0,
    dropout: float = 0.0,           # 分类头 Dropout
    # ---- 新增：轻量 Transformer 开关与超参 ----
    use_lite_tf: bool = True,
    lite_tf_blocks: int = 1,
    lite_tf_dim: int = 192,         # 建议: 128/160/192
    lite_tf_heads: int = 3,
    lite_tf_mlp_ratio: float = 1.5,
    lite_tf_attn_drop: float = 0.0,
    lite_tf_proj_drop: float = 0.0,
    lite_tf_drop_path: float = 0.1,
) -> nn.Module:
    assert 0.0 <= float(dropout) < 1.0, "dropout 应在 [0,1) 区间"

    weights = _get_cnext_tiny_weights(pretrained)
    m: tv.ConvNeXt = tv.convnext_tiny(weights=weights, drop_path_rate=drop_path_rate)

    # 1) 首层 3->13
    stem_conv: nn.Conv2d = m.features[0][0]
    stem_conv = _inflate_first_conv(stem_conv, in_ch)
    if small_stride:
        stem_conv.stride = (2, 2)  # 原 4->2
    m.features[0][0] = stem_conv

    # 2) 最后一次下采样改为 stride=1（总下采样从 16/32 降到 8）
    if small_stride:
        if isinstance(m.features[6], nn.Sequential) and len(m.features[6]) >= 2:
            down3_conv = m.features[6][1]
            if isinstance(down3_conv, nn.Conv2d):
                down3_conv.stride = (1, 1)
                m.features[6][1] = down3_conv

    # 3) 分类头：加入 Dropout（保留官方 avgpool）
    in_feats = m.classifier[2].in_features  # 768
    m.classifier = nn.Sequential(
        nn.Flatten(1),
        nn.LayerNorm(in_feats, eps=1e-6),
        nn.Dropout(p=float(dropout)),
        nn.Linear(in_feats, num_classes, bias=True),
    )

    # 4) 在 avgpool 之前串入若干个轻量 Transformer Block
    lite = nn.Identity()
    if use_lite_tf:
        blocks = []
        for _ in range(int(lite_tf_blocks)):
            blocks.append(LiteTransformerBlock(
                in_ch=in_feats,            # ConvNeXt-Tiny 最后通道数=768
                attn_dim=lite_tf_dim,
                num_heads=lite_tf_heads,
                mlp_ratio=lite_tf_mlp_ratio,
                attn_dropout=lite_tf_attn_drop,
                proj_dropout=lite_tf_proj_drop,
                drop_path=lite_tf_drop_path,
            ))
        lite = nn.Sequential(*blocks)

    # 5) 包一层 wrapper，按官方 forward 流程插入 lite
    class ConvNeXtTinyLite(nn.Module):
        def __init__(self, base: tv.ConvNeXt, lite_block: nn.Module):
            super().__init__()
            self.features   = base.features
            self.avgpool    = base.avgpool
            self.classifier = base.classifier
            self.lite       = lite_block
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)          # (N, 768, H, W)   ~ 3×3
            x = self.lite(x)              # 轻量 Transformer
            x = self.avgpool(x)           # → (N, 768, 1, 1)
            x = self.classifier(x)        # → (N, num_classes)
            return x

    return ConvNeXtTinyLite(m, lite)


# 可选：根据 patch size 自动选择 small_stride
def convnext_tiny_13ch_for_patch(
    patch_size: int,
    num_classes: int = 2,
    in_ch: int = 13,
    pretrained: bool = True,
    drop_path_rate: float = 0.0,
    dropout: float = 0.0,
) -> nn.Module:
    """
    便捷构造器：按补丁尺寸选择是否 small_stride。
    - patch <= 47: 使用 small_stride=True（总下采样=8，保留空间）
    - patch >= 48: small_stride=False（保持官方步幅）
    """
    small = patch_size <= 47
    return convnext_tiny_13ch_31(
        num_classes=num_classes,
        in_ch=in_ch,
        pretrained=pretrained,
        small_stride=small,
        drop_path_rate=drop_path_rate,
        dropout=dropout,
    )
