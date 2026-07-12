# -*- coding: utf-8 -*-
"""
Data augmentation and normalization transforms for 13-channel landslide patches.
"""
import numpy as np
import torch


def _is_chw(arr, C_expected=13):
    return isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[0] == C_expected


def _to_tensor_chw(x):
    if isinstance(x, torch.Tensor):
        t = x.float()
        if t.ndim == 3 and t.shape[0] not in (13,) and t.shape[2] == 13:
            t = t.permute(2, 0, 1).contiguous()
        return t
    a = x.astype(np.float32, copy=False)
    if a.ndim != 3:
        raise ValueError("Expect 3D array")
    if a.shape[0] != 13 and a.shape[2] == 13:
        a = np.transpose(a, (2, 0, 1))
    return torch.from_numpy(a)


class RandFlipHV:
    def __init__(self, p_h=0.5, p_v=0.5, C=13):
        self.p_h, self.p_v, self.C = p_h, p_v, C

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            H_dim, W_dim = (1, 2) if _is_chw(img, self.C) else (0, 1)
            if np.random.rand() < self.p_h:
                img = np.flip(img, axis=W_dim).copy()
            if np.random.rand() < self.p_v:
                img = np.flip(img, axis=H_dim).copy()
            return img
        if img.ndim == 3 and img.shape[0] != 13 and img.shape[2] == 13:
            img = img.permute(2, 0, 1).contiguous()
        if torch.rand(1) < self.p_h:
            img = torch.flip(img, [2])
        if torch.rand(1) < self.p_v:
            img = torch.flip(img, [1])
        return img


class RandRotate90:
    def __init__(self, C=13):
        self.C = C

    def __call__(self, img):
        k = np.random.randint(0, 4)
        if k == 0:
            return img
        if isinstance(img, np.ndarray):
            return np.rot90(img, k=k, axes=(1, 2)).copy() if _is_chw(img, self.C) \
                else np.rot90(img, k=k, axes=(0, 1)).copy()
        if img.ndim == 3 and img.shape[0] != 13 and img.shape[2] == 13:
            img = img.permute(2, 0, 1).contiguous()
        return torch.rot90(img, k=k, dims=(1, 2))


class AddGaussianNoise:
    def __init__(self, std=0.01):
        self.std = float(std)

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            return img + np.random.normal(0, self.std, size=img.shape).astype(np.float32)
        return img + self.std * torch.randn_like(img)


class NormalizeTensor:
    def __init__(self, m, s):
        self.m, self.s = m, s

    def __call__(self, img):
        return (_to_tensor_chw(img) - self.m) / self.s


class Compose:
    def __init__(self, ops):
        self.ops = ops

    def __call__(self, x):
        for op in self.ops:
            x = op(x)
        return x


def make_transforms(mean_t, std_t, noise_std=0.01):
    """Build train and validation transform pipelines."""
    train_tf = Compose([
        RandFlipHV(0.5, 0.5, 13),
        RandRotate90(13),
        AddGaussianNoise(noise_std),
        NormalizeTensor(mean_t, std_t),
    ])
    val_tf = Compose([NormalizeTensor(mean_t, std_t)])
    return train_tf, val_tf
