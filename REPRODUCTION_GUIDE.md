# ConvNeXt-Tiny 滑坡易发性分类 —— 从零复现完全指南

> 对应代码：`src/training/train.py`、`src/data/dataset.py`、`src/models/convnext.py`、`src/training/logger.py`、`src/data/extract_patches.py`  
> 目标：理解每一条代码、每一个包、每一个类的作用，并完整复现训练过程。

---

## 1. 项目概览与复现目标

### 1.1 任务定义
本项目是一个 **遥感影像块二分类** 任务：
- **输入**：大小为 `31×31`、包含 `13` 个通道的影像块
- **标签**：`0`（非滑坡）或 `1`（滑坡）
- **模型**：改造后的 `ConvNeXt-Tiny`
- **输出**：每个样本属于滑坡类别的概率

### 1.2 代码文件关系
```
src/data/extract_patches.py     # 数据准备：从 CSV + TIF 切出 13 通道 patch
src/training/train.py           # 训练入口：组装数据、模型、训练循环、指标、日志
├── src/data/dataset.py         # 自定义 Dataset：从 .npy 文件读取 13 通道样本
├── src/models/convnext.py      # 改造后的 ConvNeXt-Tiny 模型
└── src/training/logger.py      # 论文级日志封装（简化版）
```

### 1.3 复现目标
读完本文档后，你应该能够：
1. 搭建出能运行本项目的 Python 环境
2. 理解每个 `import` 进来的包/类/函数的作用
3. 准备出符合格式要求的训练数据
4. 看懂 `Dataset`、`DataLoader`、模型、`nn.Module`、`forward` 等核心概念
5. 跑通正常训练，并看懂所有输出文件
6. 知道如何开启 Optuna 超参搜索

---

## 2. 环境准备与依赖包详解

### 2.1 推荐环境
| 组件 | 推荐版本/说明 |
|---|---|
| Python | 3.9 ~ 3.11 |
| PyTorch | 2.0+（带 CUDA 更好） |
| torchvision | 与 PyTorch 对应版本 |
| numpy | 1.23+ |
| scikit-learn | 1.2+ |
| tqdm | 任意近期版本 |
| matplotlib | 3.5+ |
| optuna | 仅超参搜索时需要 |

### 2.2 安装命令示例
```bash
# 假设你使用 conda
conda create -n landslide python=3.10
conda activate landslide

# CUDA 版本（以 CUDA 11.8 为例，请根据你的显卡驱动调整）
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# 其他依赖
pip install numpy scikit-learn tqdm matplotlib optuna rasterio pandas
```

### 2.3 代码中每个 import 的作用

#### 标准库
```python
import os, json, time, csv
from pathlib import Path
from collections import Counter
```
| 包/模块 | 作用 |
|---|---|
| `os` | 拼接路径、判断环境 |
| `json` | 读取标签文件（`train_labels.json` / `val_labels.json`） |
| `time` | 生成运行 ID、统计每 epoch 耗时 |
| `csv` | 写入训练历史 CSV |
| `pathlib.Path` | 更现代的路径操作，自动处理 `/` 和 `\` |
| `collections.Counter` | 统计训练集类别数量，判断是否启用加权采样 |

#### PyTorch 核心
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
```
| 包/模块 | 作用 |
|---|---|
| `torch` | PyTorch 核心张量计算库 |
| `torch.nn` | 神经网络层、损失函数（`CrossEntropyLoss`、`Conv2d`、`Linear` 等） |
| `torch.optim` | 优化器（`AdamW`）和学习率调度器（`CosineAnnealingLR`） |
| `DataLoader` | 批量加载数据 |
| `WeightedRandomSampler` | 类别不平衡时按权重采样 |

#### 其他第三方库
```python
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, brier_score_loss,
    roc_curve, precision_recall_curve, accuracy_score
)
```
| 包/模块 | 作用 |
|---|---|
| `tqdm` | 在命令行显示进度条 |
| `matplotlib.pyplot` | 绘制 loss/acc 曲线 |
| `sklearn.metrics` | 计算 AUC、PR-AUC、F1、Brier、ROC/PR 曲线点等 |

#### 自定义模块
```python
from src.data.dataset import LandslideDataset
from src.models.convnext import convnext_tiny_13ch_31
from src.training.logger import PaperLogger, RunConfig
```
| 模块 | 作用 |
|---|---|
| `LandslideDataset` | 自定义数据集类，读取 `.npy` 样本 |
| `convnext_tiny_13ch_31` | 改造后的 ConvNeXt-Tiny 模型构造函数 |
| `PaperLogger` / `RunConfig` | 论文级日志记录 |

#### AMP 混合精度
```python
try:
    from torch.cuda.amp import autocast as amp_autocast, GradScaler as AMPGradScaler
except Exception:
    from torch.amp import autocast as amp_autocast, GradScaler as AMPGradScaler
```
- `autocast`：在前向传播时自动选择 float16/float32
- `GradScaler`：缩放损失，防止梯度下溢
- `try/except`：兼容 PyTorch 2.0 前后 `torch.cuda.amp` 被迁移到 `torch.amp` 的变化

---

## 3. Python 语法与面向对象基础（针对本代码）

### 3.1 为什么这么多 `class`？
PyTorch 的深度学习代码大量依赖**面向对象编程（OOP）**。核心思想：
- **类（class）** = 一种数据 + 行为的封装
- **实例（instance）** = 类的一个具体对象
- **方法（method）** = 类里定义的函数
- **`__init__`** = 构造函数，创建实例时自动执行
- **`__call__`** = 让实例像函数一样被调用：`obj(x)` 等价于 `obj.__call__(x)`
- **`__len__` / `__getitem__`** = 让实例支持 `len(obj)` 和 `obj[i]`，这是自定义 Dataset 的关键

### 3.2 `nn.Module`：所有神经网络层的基类
PyTorch 中所有模型、层都要继承 `nn.Module`：
```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(13, 64, 3)
    def forward(self, x):
        return self.conv(x)
```
- 必须在 `__init__` 里调用 `super().__init__()`
- `forward()` 定义数据如何前向传播
- 训练时调用 `model(x)` 会自动执行 `model.forward(x)`

### 3.3 装饰器 `@staticmethod`
```python
@staticmethod
def collate_fn(batch):
    ...
```
- 静态方法不需要 `self`
- 可以直接通过类名调用：`LandslideDataset.collate_fn(batch)`

### 3.4 列表推导式与 `zip(*batch)`
```python
imgs, labels = zip(*batch)
return torch.stack(imgs, 0), torch.stack(labels, 0)
```
- `zip(*batch)` 是 Python 的**解压**技巧
- 假设 `batch = [(img1, lab1), (img2, lab2)]`
- `zip(*batch)` 会变成 `[(img1, img2), (lab1, lab2)]`
- `torch.stack` 把多个张量沿新维度堆叠成批次

### 3.5 `*` 在函数参数中的含义
```python
def convnext_tiny_13ch_31(..., *, small_stride: bool = True, ...):
```
- `*` 之后的参数**必须用关键字传入**，不能按位置传
- 这是为了防止参数顺序传错

### 3.6 `try/except` 与兼容性处理
代码中多处使用 `try/except` 来处理不同 PyTorch 版本或不同运行环境：
```python
try: torch.set_float32_matmul_precision("high")
except Exception: pass
```
- 如果环境支持 `tf32` 就开启，不支持也不报错

---

## 4. 数据准备：从 CSV 到 JSON + npy

> 对应代码：`src/data/extract_patches.py`  
> 目标：讲清楚“最开始只有 `Dataset/*.csv` 表格，如何变成模型能吃的 `(13, 31, 31)` 样本”。

### 4.1 最开始只有 CSV 表格

当你刚克隆仓库时，`Dataset/` 里只有若干 `.csv` 文件，例如：

```
Dataset/
├── wuping_landslide_point_area_elev_stats.csv
├── wuping_landslide_point_geom_slope_litho_stats.csv
├── wuping_landslide_ridge_distance.csv
└── ...
```

这些表格记录了**滑坡点的属性统计信息**，比如每个滑坡点的：
- 投影坐标（`x`, `y`，单位：米）
- 高程、坡度、坡向
- 面积、岩性、到断层/道路的距离等

但深度学习模型**不会直接读 CSV**，它要的是一个个固定大小的影像块（patch）。所以我们需要先从 CSV 里的坐标出发，到 13 个因子栅格（TIF）上“切”出 31×31 的小块。

### 4.2 为什么需要 `extract_patches.py`？

模型 `convnext_tiny_13ch_31` 的输入是：

```
形状：(13, 31, 31)
含义：13 个通道，每个通道 31 行 × 31 列
中心像元：对应一个“滑坡点”或“非滑坡点”
```

因此，我们需要做三件事：

1. **滑坡点（label=1）**：从 CSV 读取 `(x, y)`，在 13 个 TIF 上以该坐标为中心切出 31×31 的 patch。
2. **非滑坡点（label=0）**：不能光用滑坡点训练，否则模型只会猜“全是滑坡”。需要在研究区内随机撒点，但这些点必须**远离滑坡点**（默认 ≥ 500 m），避免和滑坡特征混在一起。
3. **保存成模型能读的文件**：每个 patch 存成一个 `.npy`，再用 JSON 记录每个文件名对应的标签。

`extract_patches.py` 就是负责完成这三件事的“数据工厂”。

### 4.3 `extract_patches.py` 运行流程概览

```
Step 1: 读取 13 个因子 TIF → 拼成 (13, H, W) 的大数组
Step 2: 读取 CSV 中的滑坡点坐标 (x, y)
Step 3: 在研究区内随机生成非滑坡点，确保距滑坡点 ≥ 500 m
Step 4: 对每个坐标，在 (13, H, W) 上切 31×31 patch，保存为 .npy
Step 5: 按 8:2 划分训练集 / 验证集，生成 train_labels.json / val_labels.json
Step 6: 用 Cohen's d 检查滑坡/非滑坡样本在各通道上的差异，验证数据质量
```

### 4.4 核心函数详解

#### `read_all_factors()`：把 13 个 TIF 读进内存

```python
def read_all_factors():
    bands = []
    for fp in FACTOR_PATHS:
        with rasterio.open(fp) as src:
            bands.append(src.read(1).astype(np.float32))
    data = np.stack(bands, axis=0)
    return data
```

- `rasterio.open(fp)`：打开一个 GeoTIFF 文件。
- `src.read(1)`：读取第 1 个波段（本项目每个 TIF 只有 1 个波段）。
- `np.stack(bands, axis=0)`：把 13 个 `(H, W)` 的二维数组沿第 0 维堆叠，得到 `(13, H, W)`。

> 小白提示：这里“通道优先”`13` 放在最前面，和 PyTorch 图像张量的 `(B, C, H, W)` 一致。

#### `read_ls_points_from_csv()`：读取滑坡点坐标

```python
def read_ls_points_from_csv(csv_path):
    coords = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                x = float(row['x'].strip())
                y = float(row['y'].strip())
                coords.append((x, y))
            except (KeyError, ValueError) as e:
                if i < 5:
                    print("  跳过行 %d: %s" % (i, e))
                continue
    return coords
```

- `csv.DictReader`：把 CSV 每一行读成字典，列名就是 key。
- 默认用 `'x'` 和 `'y'` 两列作为投影坐标。
- 如果前 5 行有解析错误会打印提示，后面默默跳过，避免中断。

#### `coord_to_rowcol()`：坐标 → 栅格行列号

```python
def coord_to_rowcol(transform, x, y):
    r, c = rowcol(transform, x, y)
    return int(r), int(c)
```

- TIF 自带一个 `transform`，它描述了“行列号 ↔ 真实地理坐标”的对应关系。
- `rasterio.transform.rowcol(transform, x, y)` 就是利用这个 transform，把 `(x, y)` 转换成数组里的行 `r`、列 `c`。

#### `extract_patch()`：切 patch（带边界填充）

```python
def extract_patch(data, row, col, pad=PAD):
    C, H, W = data.shape
    r0, r1 = row - pad, row + pad + 1
    c0, c1 = col - pad, col + pad + 1

    patch = np.zeros((C, PATCH, PATCH), dtype=np.float32)

    pr0, pr1 = max(0, r0), min(H, r1)
    pc0, pc1 = max(0, c0), min(W, c1)
    ppr0 = pr0 - r0
    ppr1 = ppr0 + (pr1 - pr0)
    ppc0 = pc0 - c0
    ppc1 = ppc0 + (c1 - c0)

    patch[:, ppr0:ppr1, ppc0:ppc1] = data[:, pr0:pr1, pc0:pc1]
    return patch
```

- `PATCH = 31`，`PAD = 15`，所以中心像元上下左右各扩 15 个像元。
- 如果中心靠近图像边缘，切出来的区域会超出原图范围。
- 代码用 `np.zeros` 先创建一个全 0 的 31×31 框，再把“有效部分”贴进去，超出的部分保持 0。这叫做**零填充（zero padding）**。

> 小白提示：零填充是为了让靠近边界的点也能被切出固定大小的 patch，避免因为边缘点而丢失样本。

#### `generate_nonls_points()`：生成非滑坡点

这是整个脚本里最需要理解的函数。

**为什么不能随机选点？**

如果完全随机选点，很可能选到离滑坡只有几米的点。那些点虽然当前没滑，但地质条件可能和滑坡点几乎一样，模型学到后会把它们也判成滑坡，导致“假阳性”。

**解决办法**：
- 在研究区范围内随机撒点。
- 每个候选点都要检查：到**所有**滑坡点的距离是否 ≥ `MIN_BUFFER_DIST`（默认 500 m）。
- 太近就扔掉，重新撒。

**加速技巧：网格分桶**

滑坡点可能有成千上万个，如果每个候选点都算一遍“到所有滑坡点的距离”，会非常慢。代码用了“网格分桶”：

1. 把研究区划分成 500 m × 500 m 的网格。
2. 每个滑坡点放进它所在的网格桶里。
3. 检查候选点时，只检查它周围几个桶里的滑坡点，而不是全部。

```python
# 把滑坡点放入网格
ls_grid = {}
for lx, ly in ls_coords:
    ci = int((lx - x_min) / cell_size)
    cj = int((ly - y_min) / cell_size)
    key = (ci, cj)
    if key not in ls_grid:
        ls_grid[key] = []
    ls_grid[key].append((lx, ly))
```

这类似于快递分拣：先按区域把包裹分桶，再找附近几个区域的包裹，不用翻遍整个仓库。

#### `main()`：串联整个流程

```python
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(SEED)

    # 1. 读因子
    data = read_all_factors()
    C, H, W = data.shape

    # 2. 读滑坡点
    ls_coords = read_ls_points_from_csv(LS_CSV)

    # 3. 生成非滑坡点
    n_target = int(len(ls_coords) * N_NONLS_RATIO)
    nonls_coords = generate_nonls_points(...)

    # 4. 切 patch 并保存
    ...

    # 5. 训练/验证划分
    train_names, val_names, train_labels, val_labels = train_test_split(...)

    # 6. 数据质量检查
    ...
```

- `N_NONLS_RATIO = 1`：非滑坡点数量 = 滑坡点数量 × 1，即 1:1 正负样本。
- `train_test_split(..., test_size=0.2, stratify=labels)`：按 8:2 划分，并保持训练集和验证集中滑坡/非滑坡比例一致（分层采样）。

### 4.5 运行 `extract_patches.py`

在 `landslide` 环境中执行：

```bash
cd C:\code\landslides
python src/data/extract_patches.py
```

运行过程中你会看到类似输出：

```
==================================================
Step 1: Reading 13 factor TIFs...
  Factor stack: (13, 18006, 12366)
  CRS: EPSG:32650

Step 2: Reading landslide points...
  Landslide points: 25893

Step 3: Generating non-landslide points (buffer >= 500 m)...
  TIF坐标范围: x=[258000.0, 412000.0], y=[2670000.0, 2810000.0]
  滑坡缓冲距离: 500 m
Generating non-LS points: 100%|████████| 25893/25893
  Generated non-landslide points: 25893

Step 4: Extracting patches...
Landslide: 100%|████████| 25893/25893
Non-landslide: 100%|████████| 25893/25893

Step 5: Splitting train/val (8:2)...

Step 6: Validating data quality...
  Cohen's d (效应量: >0.8=大, 0.5=中, 0.2=小):
    Ch 0 Elevation      : LS=   417.123  nonLS=   458.234  d=0.234 (SMALL)
    Ch 1 Slope          : LS=    21.456  nonLS=    15.234  d=0.912 (LARGE)
    ...
==================================================
Done!
  Total: 51786
  Train: 41428 (LS=20714, non=20714)
  Val:   10358 (LS=5179, non=5179)
  Output: C:\code\landslides\sample
==================================================
```

> 注意：实际运行时 `Controlling_Factors/feature/*.tif` 必须存在，否则脚本会报错找不到文件。

### 4.6 生成的目录结构

运行成功后，`sample/` 会变成这样：

```
C:\code\landslides\sample\
├── train_labels.json
├── val_labels.json
├── ls_00000.npy
├── ls_00001.npy
├── nols_00000.npy
├── nols_00001.npy
└── ...
```

### 4.7 标签文件格式

`train_labels.json` / `val_labels.json` 是一个简单的字典：

```json
{
    "ls_00000": 1,
    "nols_00000": 0,
    "ls_00001": 1,
    ...
}
```

- key：样本名（不带 `.npy` 后缀）
- value：标签，`0` 表示非滑坡，`1` 表示滑坡

### 4.8 样本文件格式

每个 `.npy` 文件是一个 `numpy` 数组：

```python
import numpy as np
x = np.load("sample/ls_00000.npy")
print(x.shape)  # 期望输出: (13, 31, 31)
print(x.dtype)  # 期望输出: float32
```

- 维度顺序：**通道优先（Channel-First, CHW）**
- `13`：通道数
- `31, 31`：空间高和宽

### 4.9 归一化参数

代码里使用的 mean/std（来自 `config.py`）：

```python
MEAN = [417.616119, 17.790653, 187.580414, -0.012278, 5.720180,
        8.259916, -0.000228, 0.000378, 6.304661, 2.466416,
        2844.403564, 0.289612, 197.655365]
STD  = [185.132370, 8.436256, 101.343697, 2.483558, 2.390866,
        3.502158, 0.003821, 0.024846, 3.022274, 1.545655,
        2511.757568, 0.090679, 196.766006]
```

- 这些值是根据**整个训练集**的 13 个通道统计出来的。
- 复现时，如果你的数据分布不同，建议重新统计。
- 统计方法：遍历所有训练样本，对每个通道求均值和标准差。

---

## 5. 数据加载模块：`src/data/dataset.py` 详解

### 5.1 完整代码
```python
class LandslideDataset(Dataset):
    def __init__(self, root, json_path, transform=None):
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
```

### 5.2 逐行解释
#### `class LandslideDataset(Dataset):`
- 继承自 `torch.utils.data.Dataset`
- 自定义数据集必须实现 `__len__` 和 `__getitem__`

#### `__init__(self, root, json_path, transform=None)`
- `root`：样本 `.npy` 文件所在目录
- `json_path`：标签文件路径
- `transform`：数据增强/预处理函数（可选）
- `json.load(f).items()`：把字典变成 `(key, value)` 元组列表
- `self.samples = list(...)`：例如 `[("sample_00001", 1), ("sample_00002", 0), ...]`

#### `__len__(self)`
- 返回样本总数
- `DataLoader` 用它知道数据集有多大

#### `__getitem__(self, idx)`
- 根据索引返回一个样本
- `name, label = self.samples[idx]`：解包出样本名和标签
- `np.load(path).astype(np.float32)`：加载并确保类型为 float32
- 如果有 `transform`，就对图像做增强/归一化
- 最后返回图像张量和标签张量

#### `collate_fn(batch)`
- `DataLoader` 会把多个样本拼成一个 batch
- 默认的 collate 对张量有效，但为了明确和兼容，这里自定义
- `zip(*batch)`：把 `[(img1, lab1), (img2, lab2), ...]` 变成 `(img1, img2, ...)` 和 `(lab1, lab2, ...)`
- `torch.stack(imgs, 0)`：在第 0 维堆叠，形成 `(B, C, H, W)`
- `torch.stack(labels, 0)`：形成 `(B,)`

---

## 6. 数据增强模块：`src/data/transforms.py` 中的 transform 类

### 6.1 辅助函数
```python
def _is_chw(arr, C_expected=13):
    return isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[0] == C_expected
```
- 判断数组是否是 CHW 格式且通道数正确

```python
def _to_tensor_chw(x):
    if isinstance(x, torch.Tensor):
        t = x.float()
        if t.ndim == 3 and t.shape[0] not in (13,) and t.shape[2] == 13:
            t = t.permute(2, 0, 1).contiguous()
        return t
    a = x.astype(np.float32, copy=False)
    if a.shape[0] != 13 and a.shape[2] == 13:
        a = np.transpose(a, (2, 0, 1))
    return torch.from_numpy(a)
```
- 把 numpy 数组或张量转成 CHW 的 `torch.Tensor`
- 兼容 HWC 输入，自动转置

### 6.2 `RandFlipHV`：随机翻转
```python
class RandFlipHV:
    def __init__(self, p_h=0.5, p_v=0.5, C=13):
        self.p_h, self.p_v, self.C = p_h, p_v, C

    def __call__(self, img):
        # numpy 分支
        if isinstance(img, np.ndarray):
            H_dim, W_dim = (1, 2) if _is_chw(img, self.C) else (0, 1)
            if np.random.rand() < self.p_h:
                img = np.flip(img, axis=W_dim).copy()
            if np.random.rand() < self.p_v:
                img = np.flip(img, axis=H_dim).copy()
            return img
        # torch 分支
        if img.ndim == 3 and img.shape[0] != 13 and img.shape[2] == 13:
            img = img.permute(2, 0, 1).contiguous()
        if torch.rand(1) < self.p_h:
            img = torch.flip(img, [2])
        if torch.rand(1) < self.p_v:
            img = torch.flip(img, [1])
        return img
```
- 随机水平翻转（沿 W 轴）和垂直翻转（沿 H 轴）
- 每个轴独立以 `p_h` / `p_v` 概率触发
- 对 numpy 和 torch 分别处理

### 6.3 `RandRotate90`：随机 90 度旋转
```python
class RandRotate90:
    def __init__(self, C=13): self.C = C
    def __call__(self, img):
        k = np.random.randint(0, 4)  # 0, 1, 2, 3 分别代表 0/90/180/270 度
        ...
```
- `np.rot90(img, k=k, axes=...)` 实现旋转
- 遥感数据旋转 90 度不会破坏语义，是常用增强

### 6.4 `AddGaussianNoise`：加高斯噪声
```python
class AddGaussianNoise:
    def __init__(self, std=0.01): self.std = float(std)
    def __call__(self, img):
        return (img + np.random.normal(0, self.std, size=img.shape).astype(np.float32)) \
               if isinstance(img, np.ndarray) \
               else (img + self.std * torch.randn_like(img))
```
- 给图像加上均值为 0、标准差为 `std` 的高斯噪声
- 提升模型对噪声的鲁棒性

### 6.5 `NormalizeTensor`：标准化
```python
class NormalizeTensor:
    def __init__(self, m, s): self.m, self.s = m, s
    def __call__(self, img):
        return (_to_tensor_chw(img) - self.m) / self.s
```
- `m` 和 `s` 是 `[C, 1, 1]` 形状的张量
- 公式：`output = (input - mean) / std`
- 这是深度学习图像任务的标准预处理

### 6.6 `Compose`：组合多个变换
```python
class Compose:
    def __init__(self, ops): self.ops = ops
    def __call__(self, x):
        for op in self.ops:
            x = op(x)
        return x
```
- 把多个 transform 串联执行
- 例如：`Compose([Flip, Rotate, Noise, Normalize])`

### 6.7 训练与验证的 transform
```python
def make_transforms():
    train_tf = Compose([
        RandFlipHV(0.5, 0.5, 13),
        RandRotate90(13),
        AddGaussianNoise(0.01),
        NormalizeTensor(MEAN_T, STD_T)
    ])
    val_tf = Compose([NormalizeTensor(MEAN_T, STD_T)])
    return train_tf, val_tf
```
- 训练集：翻转 + 旋转 + 噪声 + 归一化
- 验证集：只做归一化（不能增强，否则无法公平评估）

---

## 7. 模型架构：`src/models/convnext.py` 详解

### 7.1 设计思路
原始 `torchvision.models.convnext_tiny` 是为 ImageNet 设计的：
- 输入通道：3（RGB）
- 输入尺寸：224×224
- 下采样倍数：32

本项目需要：
- 输入通道：13（多光谱/多特征）
- 输入尺寸：31×31
- 如果保持原 stride，31 经过 32 倍下采样后会变成 0，所以必须减小 stride

### 7.2 `LayerNorm2d`
```python
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
```
- 对 4D 张量 `(N, C, H, W)` 在通道维度做 LayerNorm
- 先 permute 成 `(N, H, W, C)`，再 `LayerNorm(C)`，最后 permute 回来

### 7.3 `DropPath`
```python
class DropPath(nn.Module):
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
```
- **DropPath / Stochastic Depth**：随机丢弃整个残差分支
- 只在训练时生效（`self.training` 为 True）
- 通过 `mask.floor_()` 生成 0/1 掩码
- `x.div(keep)` 是**反向缩放**，保证期望不变

### 7.4 `LiteTransformerBlock`
这是一个在 ConvNeXt 后段插入的轻量 Transformer 模块。

#### 结构
```
输入 x (B, C, H, W)
    │
    ├── 位置编码：DWConv 3×3 ──┐
    │                          ▼
    │                        x = x + pos(x)
    │                          │
    ├── LayerNorm ──► (B, H*W, C) ──► Q/K/V 投影到 attn_dim
    │                                  │
    │                            MultiheadAttention
    │                                  │
    │                            投影回 C，reshape 回 (B, C, H, W)
    │                                  │
    ▼                                x = x + DropPath(...)
    │
    ├── LayerNorm ──► MLP (1×1 conv + GELU + 1×1 conv)
    │                     │
    ▼                     ▼
                      x = x + DropPath(...)
```

#### 关键代码
```python
self.q = nn.Linear(in_ch, attn_dim)
self.k = nn.Linear(in_ch, attn_dim)
self.v = nn.Linear(in_ch, attn_dim)
self.attn = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True)
```
- 把通道维度 `in_ch`（如 768）投影到更小的 `attn_dim`（如 192），控制计算量
- `batch_first=True`：输入形状为 `(B, L, D)`

```python
y = y.permute(0, 2, 3, 1).reshape(B, H * W, C)
q, k, v = self.q(y), self.k(y), self.v(y)
z, _ = self.attn(q, k, v, need_weights=False)
```
- 把 4D 特征图拉成 token 序列，做自注意力
- `need_weights=False`：不需要返回注意力权重，节省内存

### 7.5 `_inflate_first_conv`：3→13 通道权重迁移
```python
def _inflate_first_conv(conv_rgb: nn.Conv2d, in_ch: int) -> nn.Conv2d:
    new = nn.Conv2d(
        in_channels=in_ch,
        out_channels=conv_rgb.out_channels,
        kernel_size=conv_rgb.kernel_size,
        stride=conv_rgb.stride,
        padding=conv_rgb.padding,
        bias=(conv_rgb.bias is not None),
    )
    with torch.no_grad():
        w = conv_rgb.weight
        mean_w = w.mean(dim=1, keepdim=True)          # (out, 1, k, k)
        new_w = mean_w.repeat(1, in_ch, 1, 1)         # (out, in_ch, k, k)
        c = min(3, in_ch)
        new_w[:, :c] = w[:, :c]                       # 前 3 通道拷贝原权重
        new.weight.copy_(new_w)
        if conv_rgb.bias is not None:
            new.bias.copy_(conv_rgb.bias)
    return new
```
- 创建一个新的 13 通道输入卷积
- 新通道中前 3 个通道复制 ImageNet 预训练权重
- 其余通道用 RGB 权重的均值填充
- 这样可以在多光谱数据上**部分利用**预训练知识

### 7.6 `convnext_tiny_13ch_31`：主构造函数
#### 改造步骤
1. **加载官方 ConvNeXt-Tiny**
   ```python
   weights = _get_cnext_tiny_weights(pretrained)
   m: tv.ConvNeXt = tv.convnext_tiny(weights=weights, drop_path_rate=drop_path_rate)
   ```

2. **首层 3→13 通道**
   ```python
   stem_conv: nn.Conv2d = m.features[0][0]
   stem_conv = _inflate_first_conv(stem_conv, in_ch)
   if small_stride:
       stem_conv.stride = (2, 2)  # 原 4→2
   m.features[0][0] = stem_conv
   ```

3. **减小最后一次下采样 stride**
   ```python
   if small_stride:
       down3_conv = m.features[6][1]
       down3_conv.stride = (1, 1)
       m.features[6][1] = down3_conv
   ```
   - 原总下采样 = 32
   - 修改后：stem 4→2，last downsample 2→1，总下采样 = **8**
   - 31×31 经过 8 倍下采样 → 约 3×3，保留空间信息

4. **替换分类头，加入 Dropout**
   ```python
   in_feats = m.classifier[2].in_features  # 768
   m.classifier = nn.Sequential(
       nn.Flatten(1),
       nn.LayerNorm(in_feats, eps=1e-6),
       nn.Dropout(p=float(dropout)),
       nn.Linear(in_feats, num_classes, bias=True),
   )
   ```

5. **插入 LiteTransformerBlock**
   ```python
   if use_lite_tf:
       blocks = []
       for _ in range(int(lite_tf_blocks)):
           blocks.append(LiteTransformerBlock(in_ch=in_feats, ...))
       lite = nn.Sequential(*blocks)
   ```

6. **包一层 wrapper**
   ```python
   class ConvNeXtTinyLite(nn.Module):
       def __init__(self, base: tv.ConvNeXt, lite_block: nn.Module):
           super().__init__()
           self.features   = base.features
           self.avgpool    = base.avgpool
           self.classifier = base.classifier
           self.lite       = lite_block
       def forward(self, x: torch.Tensor) -> torch.Tensor:
           x = self.features(x)
           x = self.lite(x)
           x = self.avgpool(x)
           x = self.classifier(x)
           return x
   ```
   - 这是为了在不修改 `tv.ConvNeXt` 内部结构的情况下插入 Transformer

### 7.7 前向传播数据形状变化
```
输入:          (B, 13, 31, 31)
features:      (B, 768, ~3, ~3)   # 下采样 8 倍
lite:          (B, 768, ~3, ~3)   # Transformer 不改变空间/通道尺寸
avgpool:       (B, 768, 1, 1)
Flatten:       (B, 768)
classifier:    (B, 2)
```

---

## 8. 训练流程：`src/training/train.py` 详解

### 8.1 全局配置
```python
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
torch.backends.cudnn.benchmark = True
try: torch.set_float32_matmul_precision("high")
except Exception: pass
```
- 固定随机种子，保证可复现
- `cudnn.benchmark = True`：对固定输入尺寸加速卷积
- `tf32`：在支持 Ampere 及以上架构的 GPU 上加速矩阵运算

### 8.2 `build_dataloaders`
```python
def build_dataloaders(batch_size=256, num_workers=4, prefetch_factor=2):
    train_tf, val_tf = make_transforms()
    train_set = LandslideDataset(ROOT, TRAIN_JSON, transform=train_tf)
    val_set   = LandslideDataset(ROOT, VAL_JSON,   transform=val_tf)

    # 统计类别数量
    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        train_dict = json.load(f)
    labels = list(train_dict.values())
    cnt = Counter(labels)

    # 如果类别不平衡，使用 WeightedRandomSampler
    use_sampler = (min(cnt.values()) > 0) and (max(cnt.values()) / min(cnt.values()) >= 1.1)
    if use_sampler:
        w_map = {cls: 1.0 / c for cls, c in cnt.items()}
        weights = torch.DoubleTensor([w_map[l] for l in labels])
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle_flag = False
    else:
        sampler = None
        shuffle_flag = True

    dl_kwargs = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        collate_fn=train_set.collate_fn,
        drop_last=False
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, shuffle=shuffle_flag, **dl_kwargs)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, **dl_kwargs)
    return train_loader, val_loader
```

#### 关键点
- `WeightedRandomSampler`：
  - 每个样本被采样的概率与其类别权重成正比
  - 少数类样本被重复采样，缓解类别不平衡
  - `replacement=True`：有放回采样
- `pin_memory=True`：把数据放到 CUDA 锁页内存，加速 GPU 传输
- `persistent_workers`：保持 worker 进程不销毁，减少每个 epoch 启动开销
- `drop_last=False`：最后一个 batch 不足 batch_size 也保留

### 8.3 `build_model`
```python
def build_model(dropout=0.0, **lite_kwargs):
    lite_defaults = dict(
        use_lite_tf=True,
        lite_tf_blocks=1,
        lite_tf_dim=192,
        lite_tf_heads=3,
        lite_tf_mlp_ratio=1.5,
        lite_tf_attn_drop=0.0,
        lite_tf_proj_drop=0.0,
        lite_tf_drop_path=0.1,
    )
    lite_defaults.update(lite_kwargs)

    net = convnext_tiny_13ch_31(
        num_classes=2, in_ch=13, pretrained=True, dropout=dropout, **lite_defaults
    ).to(DEVICE)
    net = net.to(memory_format=torch.channels_last)
    return net
```
- `**lite_kwargs`：接收任意关键字参数，覆盖默认值
- `to(DEVICE)`：把模型放到 GPU/CPU
- `channels_last`：内存格式优化，NVIDIA GPU 上可加速卷积

### 8.4 `run_training` 主训练函数
#### 初始化阶段
1. 创建 `run_dir`，命名格式：`YYYYMMDD-HHMMSS`
2. 写入 `run_config.json`，记录所有超参
3. 构建 DataLoader 和模型
4. 定义损失函数、优化器、学习率调度器、AMP Scaler
5. 初始化 CSV 文件头
6. 初始化 `PaperLogger`

#### 训练循环
```python
for epoch in range(1, epochs + 1):
    # 1. Warmup
    if epoch <= warmup_epochs:
        warm_lr = lr * epoch / max(1, warmup_epochs)
        for pg in optimizer.param_groups:
            pg["lr"] = warm_lr

    # 2. Train
    net.train()
    for imgs, labels in tqdm(CUDAPrefetcher(train_loader, DEVICE), ...):
        optimizer.zero_grad(set_to_none=True)
        with amp_autocast(enabled=(DEVICE.type == 'cuda')):
            outputs = net(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # 3. Valid
    net.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(CUDAPrefetcher(val_loader, DEVICE), ...):
            outputs = net(imgs)
            loss = criterion(outputs, labels)
            # 收集概率和标签

    # 4. 计算指标
    roc_auc = ...
    pr_auc = ...
    f1 = ...
    best_thr = ...  # Youden J

    # 5. 学习率调度
    if epoch > warmup_epochs:
        scheduler.step()

    # 6. 日志与保存
    # CSV 追加、PaperLogger、保存最佳模型和曲线
```

#### 关键代码解释
- `optimizer.zero_grad(set_to_none=True)`：把梯度设为 None 而不是 0， slightly 更快
- `scaler.scale(loss).backward()`：反向传播前缩放损失
- `scaler.step(optimizer)`：如果梯度正常则更新参数，否则跳过
- `scaler.update()`：更新缩放因子
- `net.train()` / `net.eval()`：控制 Dropout、BatchNorm、DropPath 等行为
- `torch.no_grad()`：验证时不计算梯度，节省显存

### 8.5 最优阈值计算
```python
fpr, tpr, thr = roc_curve(y_true, y_score)
j = tpr - fpr
k = int(np.argmax(j))
best_thr = float(thr[k])
```
- **Youden's J statistic**：`J = TPR - FPR`
- 取使 J 最大的阈值作为最优阈值
- 然后基于该阈值计算 `acc_at_best`、`tnr_at_best`

### 8.6 模型保存策略
```python
if vl_acc > best_acc:
    best_acc = vl_acc
    torch.save(net.state_dict(), best_wts)
    np.save(run_dir / "best_y_true.npy", y_true)
    np.save(run_dir / "best_y_score.npy", y_score)
    # 保存 ROC/PR 曲线点
```
- 只保存验证准确率最高时的权重
- 同时保存该 epoch 的所有概率和标签，方便后续绘图分析

### 8.7 训练结束
```python
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, tr_loss, label="train")
plt.plot(epochs_range, vl_loss, label="val")
plt.title("Loss")
plt.grid()
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(epochs_range, tr_acc, label="train")
plt.plot(epochs_range, vl_acc, label="val")
plt.title("Accuracy")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(run_dir / "loss_acc_curve.png", dpi=200)
```
- 绘制 loss 和 accuracy 曲线
- 保存为高清 PNG

---

## 9. 日志系统：`src/training/logger.py` 详解

### 9.1 `RunConfig`
```python
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
```
- `@dataclass`：自动帮你生成 `__init__`、`__repr__` 等方法
- `asdict(cfg)`：把 dataclass 转成普通字典，方便 `json.dump`

### 9.2 `PaperLogger`
```python
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
        probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
        self.val_preds.extend(probs)
        self.val_labels.extend(labels.cpu().numpy())

    def finalize_epoch(self, epoch, lr, tr_loss, tr_acc, vl_loss, vl_acc,
                       epoch_sec=0, max_mem=0, train_samples=0):
        self.val_preds = []
        self.val_labels = []
```
- 这是一个**简化版** logger，主要做配置保存
- `add_val_batch`：收集验证批次概率（虽然当前版本没有在 `finalize_epoch` 里使用，但保留了接口）
- `finalize_epoch`：每轮结束时清空缓存

---

## 10. Optuna 超参搜索

### 10.1 开关
```python
USE_OPTUNA = False
N_TRIALS = 20
TUNE_EPOCHS = 10
PRUNE_ON = True
```

### 10.2  objective 函数
```python
def objective(trial: "optuna.trial.Trial"):
    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 192, 256, 320])
    ...

    try:
        best_acc = run_training(
            epochs=TUNE_EPOCHS,
            ...,
            report_to_optuna=trial,
            prune_on=PRUNE_ON
        )
    except optuna.TrialPruned:
        raise
    return best_acc
```

### 10.3 剪枝
```python
pruner = optuna.pruners.MedianPruner(n_warmup_steps=max(2, TUNE_EPOCHS//4))
study = optuna.create_study(direction="maximize", pruner=pruner, study_name="convnext_tiny31_search")
study.optimize(objective, n_trials=N_TRIALS)
```
- `MedianPruner`：如果当前 trial 的表现低于历史 trial 的中位数，就提前剪枝
- `direction="maximize"`：目标是最大化 val_acc

---

## 11. 完整复现步骤 Checklist

### 步骤 1：准备数据（从 CSV 生成 patch）
- [ ] 确认 `Dataset/*.csv` 中存在滑坡点坐标（`x`, `y` 列）
- [ ] 确认 `Controlling_Factors/feature/` 下 13 个因子 TIF 完整
- [ ] 运行 `python src/data/extract_patches.py`，生成 `sample/*.npy` 和 `train_labels.json` / `val_labels.json`

### 步骤 2：安装环境
- [ ] 安装 Python 3.10
- [ ] 安装 PyTorch + torchvision（CUDA 版本）
- [ ] 安装 `numpy scikit-learn tqdm matplotlib optuna rasterio pandas`

### 步骤 3：检查代码路径
- [ ] 确认 `config.py` 中的 `PROJECT_ROOT`、`FACTOR_PATHS`、`LS_CSV` 指向正确位置
- [ ] 确认 `src/training/train.py` 中的 `ROOT`、`TRAIN_JSON`、`VAL_JSON` 指向正确位置
- [ ] 确认 `log_root` 和 `paper_root` 目录存在或可写

### 步骤 4：运行训练
```bash
cd C:\code\landslides
python src/training/train.py
```

### 步骤 5：观察输出
- [ ] 控制台看到 `Using cuda`
- [ ] 每个 epoch 输出 train/val loss、acc、AUC、F1 等
- [ ] 训练结束后在 `C:\code\landslides\checkpoints\YYYYMMDD-HHMMSS\` 查看结果

### 步骤 6：（可选）开启 Optuna
```python
# 在 src/training/train.py 中修改
USE_OPTUNA = True
N_TRIALS = 20
```
然后重新运行。

---

## 12. 输出文件说明

每个训练 run 会生成以下文件：

| 文件 | 说明 |
|---|---|
| `run_config.json` | 记录本次训练所有超参数和环境信息 |
| `best_convnext_tiny31_trans.pth` | 验证准确率最高时的模型权重 |
| `train_history_convnext31_trans.csv` | 每 epoch 的训练历史 |
| `loss_acc_curve.png` | loss 和 accuracy 曲线图 |
| `best_y_true.npy` | 最优 epoch 的真实标签 |
| `best_y_score.npy` | 最优 epoch 的预测概率 |
| `curves/best_ROC_points.csv` | ROC 曲线点 |
| `curves/best_PR_points.csv` | PR 曲线点 |
| `curves/best_PR_points_with_thr.csv` | 带阈值的 PR 曲线点 |

---

## 13. 常见问题与调试

### Q1：报错 `Expected 3D array`
- 检查 `.npy` 文件形状是否为 `(13, 31, 31)`
- 如果是 `(31, 31, 13)`，代码中的 `_to_tensor_chw` 会尝试转置，但 `_is_chw` 等辅助函数可能误判

### Q2：CUDA out of memory
- 减小 `batch_size`
- 减小 `lite_tf_dim` 或 `lite_tf_blocks`
- 关闭 `channels_last`（但会慢）

### Q3：训练很慢
- 确认 `DEVICE` 是 `cuda`
- 增大 `num_workers`（Windows 下可能不稳定，当前 `main()` 设为 0）
- 检查是否启用了 AMP

### Q4：val_acc 很高但 AUC/F1 很低
- 说明类别可能极度不平衡
- 考虑改用 AUC 或 F1 作为模型选择标准
- 调整 `WeightedRandomSampler` 的权重策略

### Q5：Optuna 搜索时内存爆炸
- 每个 trial 都会创建新模型，旧模型如果没有被垃圾回收可能占用显存
- 可在 trial 结束时手动 `torch.cuda.empty_cache()`

### Q6：路径硬编码导致换机器跑不了
- 把 `ROOT`、`log_root`、`paper_root` 改成命令行参数或配置文件读取
- 示例：`ROOT = os.environ.get("LANDSLIDE_ROOT", r"C:\code\landslides\sample")`

---

## 14. 关键代码片段速查

### 14.1 数据加载
```python
dataset = LandslideDataset(root, json_path, transform=transform)
loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)
```

### 14.2 模型构建
```python
model = convnext_tiny_13ch_31(
    num_classes=2,
    in_ch=13,
    pretrained=True,
    dropout=0.0,
    use_lite_tf=True,
    lite_tf_blocks=1,
    lite_tf_dim=192,
    lite_tf_heads=3
).to("cuda")
```

### 14.3 加载权重推理
```python
model = convnext_tiny_13ch_31(num_classes=2, in_ch=13)
model.load_state_dict(torch.load("best_convnext_tiny31_trans.pth"))
model.eval()

with torch.no_grad():
    x = torch.randn(1, 13, 31, 31).to("cuda")
    out = model(x)
    prob = torch.softmax(out, dim=1)[:, 1]
```

---

## 15. 总结

本文档从环境、Python 基础、数据、模型、训练、日志、复现、调试等角度完整梳理了 `ConvNeXt-Tiny 滑坡分类` 项目。核心要点：

1. **数据**：从 `Dataset/*.csv` 坐标 + `Controlling_Factors/*.tif` 因子，经 `extract_patches.py` 切出 13 通道 31×31 的 `.npy` 块 + JSON 标签
2. **模型**：基于 torchvision ConvNeXt-Tiny 改造，支持 13 通道输入，减小 stride，加入 Dropout 和 Lite Transformer
3. **训练**：AdamW + CosineAnnealing + Warmup + AMP + WeightedRandomSampler
4. **评估**：AUC、PR-AUC、F1、Brier、Youden J 最优阈值
5. **输出**：权重、CSV、曲线、ROC/PR 点、论文日志

如果你是第一次复现，建议严格按照 **第 11 节 Checklist** 逐步执行，遇到问题先回到对应章节核对。

---

## 16. ArcGIS Pro 操作基础

> 待补充：研究区数据导入、坐标系设置、TIF 可视化、CSV 点数据加载与导出、坡度/坡向/曲率等地形因子生成、距离分析等常用操作。

（本章为后续补充预留，当前为空。）

---

## 17. 工程地质分析原理

> 待补充：滑坡形成机理、降雨型群发滑坡控制因素、研究区地层岩性、构造与地貌背景、人类工程活动对斜坡稳定性的影响等。

（本章为后续补充预留，当前为空。）

