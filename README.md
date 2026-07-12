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

## 环境准备

```bash
pip install -r requirements.txt
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

默认项目根目录为 `C:\code\landslides`。可通过环境变量覆盖：

```bash
set LANDSLIDE_ROOT=C:\your\path
python src/training/train.py
```

更多细节参见 [REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md)。
