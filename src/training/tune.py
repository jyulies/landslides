# -*- coding: utf-8 -*-
"""
Optuna 超参搜索入口（独立于正式训练 train.py，互不干扰）

用法：
    python src/training/tune.py        # 20 组试验 × 10 epochs，含中位数剪枝

结果：
    - 控制台打印 best_value / best_params / 各参数重要性
    - checkpoints/optuna/trials.csv    全部试验记录
    - 每个 trial 的日志在 checkpoints/optuna/logs|paper/trial_<n>/

拿到最优参数后填入 train.py 的 main() 即可完整训练。

Reference / 引用：
    Luo, S., Mao, W., Yang, Z., Zheng, G., He, Z., Wang, J., & Huang, Y. (2026).
    CNXT-Ti-LT--Based Multi-Scale Feature--Aware Susceptibility Mapping of
    Rainfall-Induced Clustered Landslides in Southeast China.
    Journal of Geophysical Research: Machine Learning and Computation,
    3, e2025JH001115. https://doi.org/10.1029/2025JH001115
"""
import os
import sys
from pathlib import Path

# Allow imports from the project root when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import optuna
import torch

from config import CHECKPOINTS_DIR, NUM_WORKERS
from src.training.train import run_training

# === 搜索配置 ===
N_TRIALS     = 20   # 试验组数
TUNE_EPOCHS  = 10   # 每组试验训练的 epoch 数（短跑定相对优劣）
PRUNE_ON     = True # 中位数剪枝：明显差的试验提前终止


def objective(trial: "optuna.trial.Trial"):
    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 192, 256, 320])
    warmup_epochs = trial.suggest_int("warmup_epochs", 0, 5)
    label_smooth = trial.suggest_float("label_smoothing", 0.0, 0.1)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    lite_tf_blocks = trial.suggest_int("lite_tf_blocks", 0, 2)  # 0 表示关闭
    lite_tf_dim = trial.suggest_categorical("lite_tf_dim", [128, 160, 192])
    # 注意：Optuna 的 categorical 不支持动态取值空间，heads 用固定候选
    # 4 和 8 对 128/160/192 三种 dim 均可整除
    lite_tf_heads = trial.suggest_categorical("lite_tf_heads", [4, 8])
    lite_tf_mlp_ratio = trial.suggest_float("lite_tf_mlp_ratio", 1.0, 2.0)
    lite_tf_drop_path = trial.suggest_float("lite_tf_drop_path", 0.0, 0.2)
    lite_tf_attn_drop = trial.suggest_float("lite_tf_attn_drop", 0.0, 0.1)
    lite_tf_proj_drop = trial.suggest_float("lite_tf_proj_drop", 0.0, 0.1)

    model_kwargs = dict(
        use_lite_tf=(lite_tf_blocks > 0),
        lite_tf_blocks=lite_tf_blocks,
        lite_tf_dim=lite_tf_dim,
        lite_tf_heads=lite_tf_heads,
        lite_tf_mlp_ratio=lite_tf_mlp_ratio,
        lite_tf_drop_path=lite_tf_drop_path,
        lite_tf_attn_drop=lite_tf_attn_drop,
        lite_tf_proj_drop=lite_tf_proj_drop,
    )

    optuna_log_root = Path(os.environ.get(
        "OPTUNA_LOG_ROOT", str(CHECKPOINTS_DIR / "optuna" / "logs")))
    optuna_paper_root = Path(os.environ.get(
        "OPTUNA_PAPER_ROOT", str(CHECKPOINTS_DIR / "optuna" / "paper")))
    return run_training(
        epochs=TUNE_EPOCHS, warmup_epochs=warmup_epochs,
        lr=lr, weight_decay=weight_decay,
        batch_size=batch_size, num_workers=NUM_WORKERS,
        label_smoothing=label_smooth, dropout=dropout,
        model_kwargs=model_kwargs,
        log_root=str(optuna_log_root / f"trial_{trial.number}"),
        paper_root=str(optuna_paper_root / f"trial_{trial.number}"),
        report_to_optuna=trial, prune_on=PRUNE_ON
    )


def main():
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=max(2, TUNE_EPOCHS // 4))
    study = optuna.create_study(direction="maximize", pruner=pruner,
                                study_name="convnext_tiny31_search")
    study.optimize(objective, n_trials=N_TRIALS)
    print("\n=== Optuna Best ===")
    print("best_value (val_acc):", study.best_value)
    print("best_params:", study.best_params)
    imp = optuna.importance.get_param_importances(study)
    print("param_importances:", {k: round(v, 3) for k, v in imp.items()})
    # 保存全部试验结果，便于事后分析
    out_csv = CHECKPOINTS_DIR / "optuna" / "trials.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(str(out_csv), index=False)
    print("trials saved to:", out_csv)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
