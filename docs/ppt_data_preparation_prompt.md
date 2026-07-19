# 组会 PPT 生成提示词：数据准备 —— 从 CSV 到 JSON + npy

> 本提示词用于 Kimi PPT 或其他 AI PPT 工具，生成一份面向研究生组会的学术汇报幻灯片。
> 主题为 landslides 项目中 `src/data/extract_patches.py` 的数据准备流程。

---

## 一、汇报主题与目标

**主题**：数据准备 —— 从 CSV 到 JSON + npy

**副标题**：基于 ConvNeXt-Tiny 的降雨型集群式滑坡易发性制图项目

**汇报目标**：
1. 让听众理解：为什么原始 CSV 表格不能直接喂给深度学习模型。
2. 讲清楚 `extract_patches.py` 的功能由来：从地理坐标 → 影像块 → 训练样本。
3. 展示完整流程：读 TIF、读 CSV、生成非滑坡点、切 patch、划分数据集、质量检查。
4. 说明数据与后续训练的关系，为下一讲（模型训练）做铺垫。

---

## 二、目标听众

- 导师和同学
- 对深度学习或 GIS 有一定了解，但未必熟悉本项目细节
- 希望听懂"前因后果"，而不是只看代码

---

## 三、PPT 整体结构（约 18 页）

### 第 1 页：封面
- 主标题：数据准备 —— 从 CSV 到 JSON + npy
- 副标题：基于 ConvNeXt-Tiny 的降雨型集群式滑坡易发性制图
- 汇报人、日期、单位

### 第 2 页：目录
1. 从一个问题开始
2. 我们手里有什么数据
3. 模型需要什么样的输入
4. 为什么要写 extract_patches.py
5. extract_patches.py 完整流程
6. 核心函数图解
7. 运行与输出
8. 数据质量怎么保证
9. 和后续训练的衔接
10. 小结与展望

### 第 3 页：从一个问题开始
- 问题：论文给了 CSV 表格和 TIF 栅格，距离跑训练还差几步？
- 核心矛盾：表格记录的是"点"，模型需要的是"块"。
- 本讲要回答：怎么把"点"变成"块"，并且让"块"有标签。

### 第 4 页：我们手里有什么数据（1）—— CSV 表格
- 位置：`Dataset/*.csv`
- 内容示例：滑坡点的 x、y 投影坐标，以及高程、坡度、岩性、面积等统计属性
- 特点：一行一个滑坡点，是"属性表"，不是图像
- 图示：CSV 表头示意

### 第 5 页：我们手里有什么数据（2）—— TIF 因子栅格
- 位置：`Controlling_Factors/feature/*.tif`
- 13 个因子：Elevation、Slope、Aspect、TPI、Landform、TRI、Profile Curvature、Plan Curvature、TWI、Lithology、Distance2fault、NDVI、Distance2road
- 特点：每个因子是一张地理栅格图，有坐标系、仿射变换、像元大小
- 图示：栅格叠加点要素示意

### 第 6 页：CSV 和 TIF 的关系
- CSV 提供"样本中心在哪里"
- TIF 提供"样本周围长什么样"
- 二者结合，才能切出以滑坡点为中心的上下文影像块

### 第 7 页：模型需要什么样的输入
- 形状：`(13, 31, 31)`
- 含义：13 个通道 × 31 行 × 31 列
- 中心像元：对应一个样本点（滑坡或非滑坡）
- 邻域：中心上下左右各 15 个像元，共 31×31 的上下文
- 标签：0（非滑坡）或 1（滑坡）

### 第 8 页：为什么要写 extract_patches.py（1）—— 坐标转换
- 模型不会读地理坐标
- 需要把 CSV 里的 `(x, y)` 通过 TIF 的 `transform` 转成数组里的 `(row, col)`
- 再用这个行列号去切 patch

### 第 9 页：为什么要写 extract_patches.py（2）—— 负样本生成
- 训练需要两类样本：滑坡（正）和非滑坡（负）
- 不能光用滑坡点，否则模型只会输出 1
- 非滑坡点不能乱选，必须远离滑坡点，否则地质条件可能和滑坡点太像
- 本项目设置 500m 缓冲区

### 第 10 页：为什么要写 extract_patches.py（3）—— 统一存储格式
- 输出 `.npy`：二进制数组，加载快、占用小
- 输出 `train_labels.json` / `val_labels.json`：文件名到标签的映射
- 这样 `src/data/dataset.py` 才能按文件名批量读取

### 第 11 页：extract_patches.py 完整流程图
- Step 1：读取 13 个 TIF → `(13, H, W)`
- Step 2：读取 CSV 滑坡点坐标
- Step 3：随机生成非滑坡点（≥500m）
- Step 4：对每个坐标切 31×31 patch → `.npy`
- Step 5：8:2 分层划分 train/val → `.json`
- Step 6：Cohen's d 数据质量检查

### 第 12 页：核心函数 ① —— 读取数据
- `read_all_factors()`：rasterio 读 TIF，`np.stack` 成 `(13, H, W)`
- `read_ls_points_from_csv()`：读取 CSV 的 x、y 列
- `coord_to_rowcol()`：投影坐标 → 栅格行列号
- 配图：坐标转换示意图

### 第 13 页：核心函数 ② —— 切 patch 与边界填充
- `extract_patch(data, row, col, pad=15)`
- 以 `(row, col)` 为中心，切 31×31
- 边界处用零填充，保证输出形状固定
- 配图：中心点、邻域、零填充

### 第 14 页：核心函数 ③ —— 非滑坡点生成与网格分桶
- `generate_nonls_points(transform, H, W, ls_coords, n_target, min_dist, seed)`
- 500m 缓冲区：避免负样本和正样本混在一起
- 网格分桶：把滑坡点按 500m 网格分桶，只检查附近桶，显著加速
- 配图：缓冲区、网格分桶

### 第 15 页：运行命令与输出
- 命令：
  ```bash
  cd C:\code\landslides
  python src/data/extract_patches.py
  ```
- 输出目录：`sample/`
- 输出文件：
  - `train_labels.json`、`val_labels.json`
  - `ls_*.npy`、`nols_*.npy`
- 训练/验证集分布示例

### 第 16 页：数据质量怎么保证 —— Cohen's d
- 计算每个因子在滑坡/非滑坡样本上的均值差异
- 公式：`d = |mean_LS - mean_nonLS| / pooled_std`
- 判断标准：`>0.8` 大效应，`<0.2` 小效应
- 示例：Slope、TWI、Dist2road 通常为大效应

### 第 17 页：和后续训练的衔接
- 本步输出 → 下一步 `src/data/dataset.py` 读取
- `src/training/train.py` 用 DataLoader 批量加载
- 数据准备是"地基"，决定模型能不能学到东西

### 第 18 页：小结
- 原始数据：CSV 点 + TIF 栅格
- 中间桥梁：`extract_patches.py`
- 最终产物：`(13, 31, 31)` 的 `.npy` + JSON 标签
- 下一步：进入模型训练

---

## 四、预留章节（后续补充）

以下内容不在本次 PPT 中展开，但在整体课程/组会系列中需要留出位置：

- **ArcGIS Pro 操作基础**：研究区数据导入、坐标系设置、CSV 转点、TIF 因子可视化、地形因子生成、距离分析、与 Python 的衔接。
- **工程地质分析原理**：降雨型集群式滑坡机理、武平 2024 年 6·16 事件背景、13 个控制因子的地质依据、多因子耦合与易发性概念。

---

## 五、视觉与输出要求

- 风格：学术组会，白色/浅灰背景，蓝色或深灰主色
- 每页文字精简，多用 bullet、流程图、示意图
- 关键代码用等宽字体、浅色代码块
- 输出：可直接导入 Kimi PPT 的 Markdown 大纲，或逐页标题+正文+备注
