# Landslide Susceptibility Mapping with ConvNeXt-Tiny

基于 ConvNeXt-Tiny 的滑坡易发性深度学习分类项目。

## 目录结构

```
.
├── config.py                  # 集中配置：路径、归一化参数、超参
├── requirements.txt           # Python 依赖
├── src/                       # 核心源码
│   ├── data/
│   │   ├── extract_patches.py # 从 CSV + TIF 提取训练 patch
│   │   ├── dataset.py         # PyTorch Dataset
│   │   └── transforms.py      # 数据增强与归一化
│   ├── models/
│   │   └── convnext.py        # ConvNeXt-Tiny 13 通道改造模型
│   ├── training/
│   │   ├── train.py           # 训练入口
│   │   ├── logger.py          # 论文级日志
│   │   └── diagnose.py        # 数据/模型诊断
│   └── inference/
│       └── predict.py         # 全图推理
├── sample/                    # 训练样本 (.npy) 与标签 (.json)
├── checkpoints/               # 模型权重与训练日志
├── logs/                      # 论文级日志
├── output/                    # 推理概率图
├── Dataset/                   # 原始表格数据
└── Controlling_Factors/       # 13 个因子栅格 (TIF)
```

## 快速开始

### 1. 克隆仓库

打开终端，选择你希望存放项目的目录，执行：

```bash
git clone https://github.com/jyulies/landslides.git
```

克隆完成后，进入项目根目录：

```bash
cd landslides
```

> **项目根目录**是指包含 `config.py`、`requirements.txt`、`src/` 等文件和文件夹的顶层目录。例如：`C:\code\landslides` 或 `D:\projects\landslides`。

### 2. 项目根目录说明

本项目默认将 `C:\code\landslides` 作为根目录，所有数据路径（如 `sample/`、`Dataset/`、`Controlling_Factors/`）都相对于该根目录解析。

如果你把项目放到了其他位置，有两种方式指定新的根目录：

**方式一：设置环境变量（推荐）**

```bash
# Windows CMD
set LANDSLIDE_ROOT=C:\your\path\landslides

# Windows PowerShell
$env:LANDSLIDE_ROOT="C:\your\path\landslides"
```

**方式二：修改 `config.py`**

打开 `config.py`，将默认路径改为你的实际路径：

```python
PROJECT_ROOT = Path(r"C:\your\path\landslides")
```

## 环境准备

### 1. 安装 Anaconda

1. 访问 [https://www.anaconda.com/download](https://www.anaconda.com/download)
2. 下载适合你操作系统的 Anaconda Distribution 安装包（推荐 Python 3.10/3.11 版本）
3. 运行安装程序，按提示完成安装（Windows 建议勾选 "Add Anaconda to my PATH environment variable"）

详细的 Anaconda 安装与使用手册可参考 [Anaconda 官方中文用户指南](https://anaconda.org.cn/anaconda/user-guide/)。

### 2. 创建并激活 landslide 环境

打开终端（Windows 上可以使用 **Anaconda Prompt**），执行以下命令：
> 更推荐 Anaconda PowerShell Prompt <- 这个更符合linux 用户习惯

```bash
# 创建名为 landslide 的 Python 3.10 环境
conda create -n landslide python=3.10 -y

# 激活环境
conda activate landslide
```

### 3. 安装项目依赖

在激活的 `landslide` 环境中，进入项目根目录并安装依赖：

```bash
cd C:\code\landslides
pip install -r requirements.txt
```

之后所有脚本都应在 `landslide` 环境中运行：

```bash
conda activate landslide
python src/training/train.py
```

## 使用流程

1. **提取训练样本**
   ```bash
   python src/data/extract_patches.py
   ```

2. **训练模型**
   ```bash
   python src/training/train.py
   ```

3. **全图推理**
   ```bash
   python src/inference/predict.py
   ```

## 配置

默认项目根目录为 `C:\code\landslides`。若项目存放路径不同，请参考 [项目根目录说明](#2-项目根目录说明) 进行修改。

更多细节参见 [REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md)。
