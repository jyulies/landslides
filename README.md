# Landslide Susceptibility Mapping with ConvNeXt-Tiny (CNXT-Ti-LT)

> 本仓库是论文 **“CNXT-Ti-LT–Based Multi-Scale Feature–Aware Susceptibility Mapping of Rainfall-Induced Clustered Landslides in Southeast China”** 的参考实现。
>
> - **作者**：Senlin Luo, Wuwei Mao, Zhiqiang Yang, Guoming Zheng, Zhonghui He, Jiahuang Wang, Yu Huang  
> - **期刊**：*Journal of Geophysical Research: Machine Learning and Computation*, 3, e2025JH001115  
> - **DOI**：[10.1029/2025JH001115](https://doi.org/10.1029/2025JH001115)  
> - **通讯作者**：Y. Huang ([yhuang@tongji.edu.cn](mailto:yhuang@tongji.edu.cn))  
> - **研究区**：2024 年 6 月 16 日强降雨诱发集群式滑坡的福建武平及周边粤闽赣交界区。

基于 ConvNeXt-Tiny 与 Lite-Transformer 的滑坡易发性深度学习分类项目。

## 论文基础与模型说明

本研究针对 2024 年 6 月 16 日华南极端降雨诱发的集群式滑坡事件，以福建武平县为核心研究区，构建了一套融合 13 个易发性因子的深度学习制图流程。代码中的网络结构、数据预处理与推理方式均对应论文方法。

### CNXT-Ti-LT 模型

- **输入**：31 × 31 像素、13 通道的因子 patch（中心像元对应一个滑坡点或非滑坡点）。
- **卷积骨干**：ConvNeXt-Tiny，通过大核深度可分离卷积、倒残差瓶颈与分层下采样扩大有效感受野，提取地形纹理与地貌语义。
- **多尺度融合**：在 neck 处加入 Feature Pyramid（FPN）与 skip connection，缓解边界模糊与小斑块信息损失。
- **全局语义分支**：在 Stage-4 后接入 Lite-Transformer，利用多头自注意力（MHSA）刻画 13 个因子之间的长距离结构关系。
- **输出**：二分类 Softmax 概率，即该 patch 中心像元发生滑坡的易发性。

### 13 个易发性因子

代码中 `config.py` 的 `FACTOR_PATHS` 顺序必须与本表一致。

| 编号 | 因子 | 说明 / 来源 |
| :--- | :--- | :--- |
| 1 | Elevation | 高程（m），ALOS 12.5 m DEM |
| 2 | Slope | 坡度（°），ALOS 12.5 m DEM |
| 3 | Aspect | 坡向（°），ALOS 12.5 m DEM |
| 4 | Topographic Position Index (TPI) | 地形位置指数，ALOS 12.5 m DEM |
| 5 | Landform Class | 基于 TPI 的地形分类，ALOS 12.5 m DEM |
| 6 | Terrain Ruggedness Index (TRI) | 地形粗糙度指数，ALOS 12.5 m DEM |
| 7 | Profile Curvature | 剖面曲率，ALOS 12.5 m DEM |
| 8 | Plan Curvature | 平面曲率，ALOS 12.5 m DEM |
| 9 | Topographic Wetness Index (TWI) | 地形湿度指数，ALOS 12.5 m DEM |
| 10 | Lithology | 岩性类型，GliM 全球岩性数据库 |
| 11 | Distance to Fault (DTF) | 距断层距离（m），全国 1:20 万地质图 |
| 12 | NDVI | 归一化植被指数，Sentinel-2 10 m |
| 13 | Distance to Road (DTR) | 距道路距离（m），吉林一号影像提取的道路网 |

### 数据与代码来源

本仓库的实现与数据对应作者在原论文中发布的 Zenodo 资源：

| 资源 | Zenodo 记录 | DOI | 内容说明 |
| :--- | :--- | :--- | :--- |
| 统计数据集 | [zenodo.org/records/18226013](https://zenodo.org/records/18226013) | [10.5281/zenodo.18226013](https://doi.org/10.5281/zenodo.18226013) | 武平降雨型集群式滑坡统计表格数据 |
| 训练样本与因子 | [zenodo.org/records/18463063](https://zenodo.org/records/18463063) | [10.5281/zenodo.18463063](https://doi.org/10.5281/zenodo.18463063) | 用于 CNXT-Ti-LT 易发性制图的训练数据集（含 13 个因子栅格） |
| 模型代码 | [zenodo.org/records/17509051](https://zenodo.org/records/17509051) | [10.5281/zenodo.17509051](https://doi.org/10.5281/zenodo.17509051) | CNXT-Ti-LT 模型官方参考代码 |

### 引用

如果你在研究中使用了本仓库的代码或模型，请引用论文：

```bibtex
@article{luo2026cnxt,
  title={CNXT-Ti-LT--Based Multi-Scale Feature--Aware Susceptibility Mapping of Rainfall-Induced Clustered Landslides in Southeast China},
  author={Luo, Senlin and Mao, Wuwei and Yang, Zhiqiang and Zheng, Guoming and He, Zhonghui and Wang, Jiahuang and Huang, Yu},
  journal={Journal of Geophysical Research: Machine Learning and Computation},
  volume={3},
  pages={e2025JH001115},
  year={2026},
  publisher={Wiley},
  doi={10.1029/2025JH001115}
}
```

若使用了作者发布的原始数据或官方代码，也请同时引用对应的 Zenodo 记录：

```bibtex
@dataset{luo2026wupingdata,
  title={Wuping rainfall clustered landslides statistical data},
  author={Luo, Senlin},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.18226013}
}

@dataset{luo2026training,
  title={Training Dataset for {CNXT-Ti-LT} Landslide Susceptibility Mapping},
  author={Luo, Senlin and Huang, Yu and Mao, Wuwei and Yang, Zhiqiang},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.18463063}
}

@software{luo2026cnxtcode,
  title={{CNXT-Ti-LT} model code},
  author={Luo, Senlin and Huang, Yu and Mao, Wuwei and Yang, Zhiqiang},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.17509051}
}
```

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
