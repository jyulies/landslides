# 重庆永川区滑坡易发性制图（CNXT-Ti-LT，16 通道）

基于 ConvNeXt-Tiny + Lite-Transformer 的滑坡易发性深度学习制图项目，研究区为**重庆市永川区**。

模型方法参考论文
**"CNXT-Ti-LT–Based Multi-Scale Feature–Aware Susceptibility
Mapping of Rainfall-Induced Clustered Landslides in Southeast
China"**（Luo et al., 2026, *JGR: Machine Learning and Computation*,
[10.1029/2025JH001115](https://doi.org/10.1029/2025JH001115)），
在其 13 因子体系上扩展了 3 个降雨因子。

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
- **因子栅格**：`Dataset/yonchuan/factor_aligned/`，
16 个已对齐到 30m 网格的 TIF（随仓库分发，约 30MB；已完成 nodata 清洗与单位修正）
- **样本标签/采样点**：`sample_yongchuan/` 下的 `train/val_labels.json` 与 `train/val_points.json`

**采样策略**（`src/data/extract_patches.py`）：

- **正样本（8021 个中 4055 个）**：每个滑坡多边形内取"像元中心严格落在面内"的像元，
  贪心最远点选取、上限 5 个/面；无内部像元的小面（577 个）回退到质心。
  每个点附带安全抖动半径 `j`（距离变换计算，保底 ±1 像元 = 编目定位误差量级），
  训练时 `RasterPatchDataset` 每个 epoch 在 ±j 内随机偏移 patch 中心（数据增强）
- **非滑坡点**：全图随机采样，避开滑坡面覆盖的像元，数量与正样本 1:1
- **train/val 划分**：正样本**按多边形** 8:2 划分（同一面的所有点进同一侧，防空间泄漏）；
  非滑坡点随机 8:2

> 仓库自带因子栅格与采样点，**克隆后可直接训练**（内存直读模式，
> `train.py` 中 `USE_RASTER_DATA = True`：因子一次读入内存约 31MB，
> 训练时现切 patch，无需生成 .npy 样本文件）。

## 目录结构

```palin
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

方式一（推荐，一条命令；rasterio 走 conda-forge 自带 GDAL，避免源码编译失败）：

```bash
conda env create -f environment.yml
conda activate landslide
```

方式二（手动）：

```bash
conda create -n landslide python=3.10 -y
conda activate landslide
conda install -c conda-forge rasterio scipy -y   # 地理类库用 conda 装
pip install -r requirements.txt
```

> ⚠️ 不要直接 `pip install rasterio`：部分 Linux 环境没有预编译 wheel，
> 会因系统 GDAL 版本过低编译失败。务必用 conda-forge 安装。
> GPU 版 PyTorch 如未被正确安装，请按 [pytorch.org](https://pytorch.org)
> 提供的命令单独安装（匹配服务器的 CUDA 版本）。

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

# 滑坡多边形 -> 质心点 CSV（参考产物，采样已不再依赖它）
python src/data/shp_centroids_to_csv.py

# 重新采样：面内多点采样 + 按多边形划分 train/val，生成标签 + 采样点 JSON；
# SAVE_NPY=True 时同时写 .npy（供 USE_RASTER_DATA=False 的旧模式）
python src/data/extract_patches.py
```

## 跨区域直接迁移（用永川模型预测新区）

前提（不满足则迁移结果无意义）：① 16 个因子的单位/定义与永川一致；
② 岩性、地貌的**类别编码与永川同图例**；③ 降雨三因子对应新区滑坡事件的同期降雨。

```bash
# 1. 新区因子放入 Dataset/<新区>/factor/（可直接用标准命名：Elevation.tif、Slope.tif、
#    DailyMax.tif 等；标准命名文件视为已清洗——单位米、nodata 规范）
# 2. 对齐到新区统一网格（--ref 选该区任一 30m 栅格）
python src/data/align_factors.py --src Dataset/<新区>/factor \
    --dst Dataset/<新区>/factor_aligned --ref Dataset/<新区>/factor/Elevation.tif

# 3. 直接预测（权重自动取永川最新训练结果，MEAN/STD 保持永川不变）
#    Windows: set LANDSLIDE_FACTOR_DIR=Dataset/<新区>/factor_aligned
export LANDSLIDE_FACTOR_DIR=Dataset/<新区>/factor_aligned
python src/inference/predict.py    # 输出 output/prob_<新区>.tif

# 4. 若有新区滑坡编目 shp，定量评估迁移效果
python src/inference/eval_transfer.py --prob output/prob_<新区>.tif \
    --shp Dataset/<新区>/shp/landslides.shp
```

迁移 AUC 参考：>0.8 良好；0.7~0.8 可接受；<0.7 建议用新区样本微调或重训练。

## 超参搜索

```bash
python src/training/tune.py   # Optuna 搜索：20 组试验 × 10 epochs，独立于 train.py
```

搜索学习率、正则化（dropout / weight_decay / drop_path 等）与 Lite-Transformer 结构参数。
结束后打印最优参数与各参数重要性，全部试验结果保存到 `checkpoints/optuna/trials.csv`。
将最优参数填入 `train.py` 的 `main()` 即可进行完整训练。

训练内置**早停**（`train.py` 顶部 `EARLY_STOP/ES_PATIENCE/ES_MIN_DELTA`）：
监控 val_loss，连续 12 轮无实质改善即停止（val_acc 平台化后 loss 仍会恶化，
是更敏感的过拟合信号）；best 权重仍按 val_acc 单独保存。

## 配置说明

- 所有路径由 `config.py` 自动推导（以 `config.py` 所在目录为项目根）
克隆到任意位置均可直接运行，无需修改；特殊情况可用环境变量 `LANDSLIDE_ROOT` 覆盖
- 归一化统计量 `MEAN`/`STD` 已按永川样本集、16 通道预计算；若改动因子或样本，需重算并更新 `config.py`
- 推理权重/输出路径可用环境变量 `LANDSLIDE_WEIGHTS` / `LANDSLIDE_OUTPUT` 覆盖;
  跨区域场景另支持 `LANDSLIDE_FACTOR_DIR`（因子目录）、`LANDSLIDE_SAMPLE_DIR`（样本目录）、
`LANDSLIDE_LS_SHP`（滑坡面 shp）

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
