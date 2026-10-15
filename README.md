# 重庆永川区滑坡易发性制图（CNXT-Ti-LT，16 通道）

基于 ConvNeXt-Tiny + Lite-Transformer 的滑坡易发性深度学习制图项目，研究区为**重庆市永川区**。

模型方法参考论文 **"CNXT-Ti-LT–Based Multi-Scale Feature–Aware Susceptibility Mapping of Rainfall-Induced Clustered Landslides in Southeast China"**（Luo et al., 2026, *JGR: Machine Learning and Computation*, [10.1029/2025JH001115](https://doi.org/10.1029/2025JH001115)），在其 13 因子体系上扩展了 3 个降雨因子。

## 模型

- **输入**：31 × 31 像素、16 通道因子 patch（中心像元对应一个滑坡/非滑坡样本点）
- **卷积骨干**：ConvNeXt-Tiny（首层卷积由 3 通道 ImageNet 权重扩展至 16 通道）
- **全局语义分支**：Stage-4 后可接 Lite-Transformer（MHSA 刻画因子间长距离关系）
- **输出**：二分类 Softmax 概率，即中心像元的滑坡易发性

## 16 个输入因子

`config.py` 中 `FACTOR_PATHS` 的顺序必须与本表一致（通道数由 `config.IN_CH` 统一管理）。

| 编号 | 因子 | 文件 | 说明 |
| :--- | :--- | :--- | :--- |
| 1 | Elevation | Elevation.tif | 高程（m） |
| 2 | Slope | Slope.tif | 坡度（°） |
| 3 | Aspect | Aspect.tif | 坡向（°） |
| 4 | TPI | TPI.tif | 地形位置指数 |
| 5 | Landform class | Landform class.tif | 地貌分类（类别） |
| 6 | TRI | TRI.tif | 地形粗糙度指数 |
| 7 | Profile Curvature | Profile Curvature.tif | 剖面曲率 |
| 8 | Plan Curvature | Plan Curvature.tif | 平面曲率 |
| 9 | TWI | TWI.tif | 地形湿度指数 |
| 10 | Lithology | Lithology.tif | 岩性（类别） |
| 11 | Distance2fault | Distance2fault.tif | 距断层距离（m） |
| 12 | NDVI | NDVI.tif | 归一化植被指数 |
| 13 | Distance2road | Distance2road.tif | 距道路距离（m） |
| 14 | DailyMax | DailyMax.tif | 最大单日降雨（mm） |
| 15 | Acc3d | Acc3d.tif | 3 日累计降雨（mm） |
| 16 | Acc7d | Acc7d.tif | 7 日累计降雨（mm） |

## 数据

- **滑坡编目**：`Dataset/yonchuan/shp/landslides.shp`，2394 个滑坡多边形（EPSG:32648）
- **质心点**：`Dataset/yonchuan/yongchuan_landslide_centroids.csv`（由多边形生成，正样本采样点）
- **因子栅格**：`Dataset/yonchuan/factor_aligned/`，16 个已对齐到 30m 网格的 TIF（随仓库分发，约 30MB；已完成 nodata 清洗与单位修正）
- **样本标签/采样点**：`sample_yongchuan/` 下的 `train/val_labels.json` 与 `train/val_points.json`

> 仓库自带因子栅格与采样点，**克隆后可直接训练**（内存直读模式，
> `train.py` 中 `USE_RASTER_DATA = True`：因子一次读入内存约 31MB，
> 训练时现切 patch，无需生成 .npy 样本文件）。

## 目录结构

```
.
├── config.py                      # 集中配置：路径、16 因子、归一化统计量、超参
├── requirements.txt
├── src/
│   ├── data/
│   │   ├── align_factors.py       # 因子栅格重投影到 30m 统一网格 + nodata 清洗
│   │   ├── shp_centroids_to_csv.py# 滑坡多边形 -> 质心点 CSV
│   │   ├── extract_patches.py     # 提取 31x31 patch 训练样本
│   │   ├── dataset.py             # PyTorch Dataset
│   │   └── transforms.py          # 数据增强与归一化
│   ├── models/
│   │   └── convnext.py            # ConvNeXt-Tiny 多通道改造 + Lite-Transformer
│   ├── training/
│   │   ├── train.py               # 训练入口（支持 Optuna 超参搜索）
│   │   ├── logger.py              # 论文级日志
│   │   └── diagnose.py            # 数据/模型诊断
│   └── inference/
│       └── predict.py             # 全图推理（瓦片 + AMP）
├── Dataset/yonchuan/              # 滑坡 shp、质心 CSV、因子栅格（TIF 不上传）
└── sample_yongchuan/              # 训练样本 (.npy 不上传) 与标签 (.json)
```

## 环境准备

```bash
conda create -n landslide python=3.10 -y
conda activate landslide
pip install -r requirements.txt
```

## 使用流程

```bash
# 1. 训练（克隆后可直接运行：因子栅格与采样点已随仓库分发）
python src/training/train.py

# 2. 全图推理（输出滑坡概率 GeoTIFF 到 output/）
python src/inference/predict.py
```

可选步骤（仅当更换原始因子或调整采样策略时需要）：

```bash
# 因子对齐：把 Dataset/yonchuan/factor/ 下的原始 TIF 重投影到 30m 统一网格
python src/data/align_factors.py

# 滑坡多边形 -> 质心点 CSV（仓库已附带生成结果）
python src/data/shp_centroids_to_csv.py

# 重新采样：生成标签 + 采样点 JSON；SAVE_NPY=True 时同时写 .npy（供 USE_RASTER_DATA=False 的旧模式）
python src/data/extract_patches.py
```

## 超参搜索

`train.py` 内置 Optuna 搜索（`USE_OPTUNA = True` 开启），默认 20 组试验 × 10 epochs，
搜索学习率、正则化（dropout / weight_decay / drop_path 等）与 Lite-Transformer 结构参数。
结束后打印最优参数与各参数重要性，全部试验结果保存到 `checkpoints/optuna/trials.csv`。
将最优参数填入 `main()` 并设 `USE_OPTUNA = False` 即可进行完整训练。

## 配置说明

- 所有路径由 `config.py` 自动推导（以 `config.py` 所在目录为项目根），克隆到任意位置均可直接运行，无需修改；特殊情况可用环境变量 `LANDSLIDE_ROOT` 覆盖
- 归一化统计量 `MEAN`/`STD` 已按永川样本集、16 通道预计算；若改动因子或样本，需重算并更新 `config.py`
- 推理权重/输出路径可用环境变量 `LANDSLIDE_WEIGHTS` / `LANDSLIDE_OUTPUT` 覆盖

## 引用

模型方法源自以下论文，使用本仓库代码请引用：

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
