# -*- coding: utf-8 -*-
"""
ConvNeXt-Tiny (13ch, 2-class, 31x31 patch) — 修正版 + 论文级日志 + Optuna超参搜索（逻辑不变）
开关：
- USE_OPTUNA=False  → 正常训练 (main)
- USE_OPTUNA=True   → 用 Optuna 搜索若干超参，objective 内部复用同一套训练流程，但 epoch 减少用于快速评估

Reference / 引用：
    Luo, S., Mao, W., Yang, Z., Zheng, G., He, Z., Wang, J., & Huang, Y. (2026).
    CNXT-Ti-LT--Based Multi-Scale Feature--Aware Susceptibility Mapping of
    Rainfall-Induced Clustered Landslides in Southeast China.
    Journal of Geophysical Research: Machine Learning and Computation,
    3, e2025JH001115. https://doi.org/10.1029/2025JH001115

    Official model code / 官方代码: https://doi.org/10.5281/zenodo.17509051
"""
import os, sys, json, time, csv
from pathlib import Path
from collections import Counter

import numpy as np

# Allow imports from the project root when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import matplotlib.pyplot as plt

# === 指标与曲线 ===
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, brier_score_loss,
    roc_curve, precision_recall_curve, accuracy_score
)

# === Optuna 开关（原逻辑保留） ===
USE_OPTUNA   = False
N_TRIALS     = 20
TUNE_EPOCHS  = 10
PRUNE_ON     = True

from config import PROJECT_ROOT, SAMPLE_DIR, CHECKPOINTS_DIR, LOGS_DIR, MEAN, STD, SEED
from src.data.dataset import LandslideDataset
from src.data.transforms import make_transforms
from src.models.convnext import convnext_tiny_13ch_31
from src.training.logger import PaperLogger, RunConfig

# AMP 兼容导入
try:
    from torch.cuda.amp import autocast as amp_autocast, GradScaler as AMPGradScaler
except Exception:
    from torch.amp import autocast as amp_autocast, GradScaler as AMPGradScaler

# ====== 全局默认配置（作为“基线”，Optuna 会覆盖其中一部分） ======
torch.manual_seed(SEED); np.random.seed(SEED)
torch.backends.cudnn.benchmark = True
try: torch.set_float32_matmul_precision("high")
except Exception: pass

ROOT       = str(SAMPLE_DIR)
TRAIN_JSON = os.path.join(ROOT, "train_labels.json")
VAL_JSON   = os.path.join(ROOT, "val_labels.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", DEVICE)

MEAN_T = torch.tensor(MEAN, dtype=torch.float32)[:, None, None]
STD_T  = torch.tensor(STD,  dtype=torch.float32)[:, None, None]

# ===== CUDA 预取器（不改逻辑）=====
class CUDAPrefetcher:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if device.type=="cuda" else None
    def __iter__(self):
        self.it = iter(self.loader); self.next = None; self._preload(); return self
    def __next__(self):
        if self.next is None: raise StopIteration
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next; self._preload(); return batch
    def _preload(self):
        try:
            imgs, labels = next(self.it)
        except StopIteration:
            self.next = None; return
        if self.stream is None:
            self.next = (imgs.to(self.device), labels.to(self.device)); return
        with torch.cuda.stream(self.stream):
            imgs = imgs.to(self.device, non_blocking=True).to(memory_format=torch.channels_last)
            labels = labels.to(self.device, non_blocking=True)
        self.next = (imgs, labels)

# ===== 数据 + 模型 + 训练的“可复用封装” =====
def build_dataloaders(batch_size=256, num_workers=4, prefetch_factor=2):
    train_tf, val_tf = make_transforms(MEAN_T, STD_T)
    train_set = LandslideDataset(ROOT, TRAIN_JSON, transform=train_tf)
    val_set   = LandslideDataset(ROOT, VAL_JSON,   transform=val_tf)

    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        train_dict = json.load(f)
    labels = list(train_dict.values())
    cnt = Counter(labels)
    print("Train label counts:", cnt)

    use_sampler = (min(cnt.values())>0) and (max(cnt.values())/min(cnt.values()) >= 1.1)
    if use_sampler:
        w_map = {cls: 1.0/c for cls,c in cnt.items()}
        weights = torch.DoubleTensor([w_map[l] for l in labels])
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle_flag = False
        print("Using WeightedRandomSampler.")
    else:
        sampler = None; shuffle_flag = True
        print("Class balanced. Using plain shuffle.")

    dl_kwargs = dict(num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0),
                     prefetch_factor=(prefetch_factor if num_workers > 0 else None), collate_fn=train_set.collate_fn, drop_last=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, shuffle=shuffle_flag, **dl_kwargs)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, **dl_kwargs)
    return train_loader, val_loader

def build_model(
    dropout=0.0,
    **lite_kwargs
):
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

def run_training(epochs=80, warmup_epochs=3, lr=3e-4, weight_decay=1e-4,
                 batch_size=256, num_workers=4, label_smoothing=0.05, dropout=0.0,
                 model_kwargs: dict | None = None,
                 log_root=str(CHECKPOINTS_DIR),
                 paper_root=str(LOGS_DIR),
                 report_to_optuna=None,  # 传入 trial 对象即可上报
                 prune_on=False):

    # ===== 目录与 run_config =====
    save_root = Path(log_root); save_root.mkdir(parents=True, exist_ok=True)
    RUN_ID = time.strftime("%Y%m%d-%H%M%S")
    run_dir = save_root / RUN_ID
    (run_dir / "curves").mkdir(parents=True, exist_ok=True)
    best_wts = run_dir / "best_convnext_tiny31_trans.pth"

    run_config = {
        "run_id": RUN_ID, "seed": SEED, "device": str(DEVICE),
        "root": ROOT, "train_json": TRAIN_JSON, "val_json": VAL_JSON,
        "epochs": epochs, "warmup_epochs": warmup_epochs, "batch_size": batch_size,
        "optimizer": {"type":"AdamW","lr":lr,"weight_decay":weight_decay},
        "scheduler": "CosineAnnealingLR (eta_min=1e-6)",
        "label_smoothing": label_smoothing, "dropout": dropout,
        "lite_kwargs": model_kwargs or {},
        "mean": MEAN, "std": STD,
        "augment": ["flip","rot90","gauss(0.01)","normalize"],
        "torch": torch.__version__, "cuda": (torch.version.cuda or "cpu"),
        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    }
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)
    print(f"✓ run_dir = {run_dir}")

    # ===== Data & Model =====
    train_loader, val_loader = build_dataloaders(batch_size=batch_size, num_workers=num_workers)
    net = build_model(dropout=dropout, **(model_kwargs or {}))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    TMAX = max(1, epochs - warmup_epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TMAX, eta_min=1e-6)
    scaler = AMPGradScaler(enabled=(DEVICE.type == 'cuda'))

    # ===== CSV（扩展版） =====
    csv_path = run_dir / "train_history_convnext31_trans.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch","lr",
            "train_loss","train_acc",
            "val_loss","val_acc",
            "precision","recall","f1",
            "roc_auc","pr_auc","brier",
            "best_thr","tpr_at_best","fpr_at_best",
            "acc_at_best","tnr_at_best","epoch_sec","max_mem"
        ])

    # ===== 论文级 logger（原来就有，保留） =====
    cfg = RunConfig(
        seed=SEED, model="ConvNeXt-Tiny", in_ch=13, patch=31, epochs=epochs, batch=batch_size,
        lr=lr, weight_decay=weight_decay, optimizer="AdamW", scheduler="CosineAnnealing+Warmup",
        dataset_root=ROOT, train_json=TRAIN_JSON, val_json=VAL_JSON,
        mean=MEAN, std=STD, augment="flip+rot90+gauss(0.01)", device=str(DEVICE),
        torch=torch.__version__, cuda=(torch.version.cuda or "cpu"),
        cudnn=(torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else -1),
        gpu_name=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    )
    paper_dir = Path(paper_root) / "convnext_tiny31_trans"
    paper_dir.mkdir(parents=True, exist_ok=True)
    logger = PaperLogger(paper_dir, cfg)

    # ===== 训练循环（逻辑不变） =====
    best_acc = 0.0
    train_hist, val_hist = [], []
    for epoch in range(1, epochs+1):
        t0 = time.time()
        # warmup
        if epoch <= warmup_epochs:
            warm_lr = lr * epoch / max(1, warmup_epochs)
            for pg in optimizer.param_groups: pg["lr"] = warm_lr

        # ---- train ----
        net.train()
        run_loss, correct, n = 0.0, 0, 0
        for imgs, labels in tqdm(CUDAPrefetcher(train_loader, DEVICE), desc=f"Train {epoch}/{epochs}", leave=False):
            optimizer.zero_grad(set_to_none=True)
            with amp_autocast(enabled=(DEVICE.type == 'cuda')):
                outputs = net(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bs = imgs.size(0)
            run_loss += loss.item() * bs
            correct  += (outputs.argmax(1) == labels).sum().item()
            n        += bs

        tr_loss = run_loss / max(1, n)
        tr_acc  = correct / max(1, n)
        train_hist.append((tr_loss, tr_acc))

        # ---- valid ----
        net.eval(); v_loss, v_corr, vn = 0.0, 0, 0
        y_true, y_score = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(CUDAPrefetcher(val_loader, DEVICE), desc="Valid", leave=False):
                with amp_autocast(enabled=(DEVICE.type == 'cuda')):
                    outputs = net(imgs)
                    loss = criterion(outputs, labels)
                logger.add_val_batch(outputs, labels)  # 原论文统计
                bs = imgs.size(0)
                v_loss += loss.item() * bs
                v_corr += (outputs.argmax(1) == labels).sum().item()
                vn     += bs
                # 概率（正类）
                prob1 = torch.softmax(outputs, dim=1)[:, 1]
                y_true.append(labels.cpu().numpy())
                y_score.append(prob1.cpu().numpy())

        vl_loss = v_loss / max(1, vn)
        vl_acc  = v_corr / max(1, vn)
        val_hist.append((vl_loss, vl_acc))

        # ===== 计算全套指标（不改训练） =====
        y_true  = np.concatenate(y_true)
        y_score = np.concatenate(y_score)
        y_hat   = (y_score >= 0.5).astype(np.uint8)

        roc_auc = float(roc_auc_score(y_true, y_score))
        pr_auc  = float(average_precision_score(y_true, y_score))
        brier   = float(brier_score_loss(y_true, y_score))
        prec    = float(precision_score(y_true, y_hat, zero_division=0))
        recall  = float(recall_score(y_true, y_hat, zero_division=0))
        f1      = float(f1_score(y_true, y_hat, zero_division=0))

        # 最优阈值（Youden J）
        fpr, tpr, thr = roc_curve(y_true, y_score)
        j = tpr - fpr; k = int(np.argmax(j))
        best_thr    = float(thr[k])
        tpr_at_best = float(tpr[k])
        fpr_at_best = float(fpr[k])
        y_hat_best  = (y_score >= best_thr).astype(np.uint8)
        acc_at_best = float(accuracy_score(y_true, y_hat_best))
        tnr_at_best = 1.0 - fpr_at_best

        # 调度
        if epoch > warmup_epochs: scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        epoch_sec = time.time() - t0
        max_mem = int(torch.cuda.max_memory_allocated()/1024/1024) if DEVICE.type=="cuda" else 0

        # CSV 追加
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, lr_now,
                tr_loss, tr_acc,
                vl_loss, vl_acc,
                prec, recall, f1,
                roc_auc, pr_auc, brier,
                best_thr, tpr_at_best, fpr_at_best,
                acc_at_best, tnr_at_best, epoch_sec, max_mem
            ])

        # 论文日志（原有）
        logger.finalize_epoch(epoch, lr_now, tr_loss, tr_acc, vl_loss, vl_acc,
                              epoch_sec=epoch_sec, max_mem=max_mem, train_samples=n)
        if DEVICE.type=="cuda":
            torch.cuda.reset_peak_memory_stats()

        # 按原逻辑：用 val_acc 选最优，并在“最优”时落盘曲线与分数
        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(net.state_dict(), best_wts)
            # 保存最佳 epoch 的分数与曲线点
            np.save(run_dir / "best_y_true.npy",  y_true)
            np.save(run_dir / "best_y_score.npy", y_score)
            # ROC：fpr、tpr、threshold 等长
            np.savetxt(run_dir / "curves/best_ROC_points.csv",
                       np.c_[fpr, tpr, thr], delimiter=",",
                       header="fpr,tpr,threshold", comments="")
            # PR：常规只存 recall/precision；另存带阈值对齐版
            prec_c, rec_c, thr_pr = precision_recall_curve(y_true, y_score)
            np.savetxt(run_dir / "curves/best_PR_points.csv",
                       np.c_[rec_c, prec_c], delimiter=",",
                       header="recall,precision", comments="")
            np.savetxt(run_dir / "curves/best_PR_points_with_thr.csv",
                       np.c_[thr_pr, rec_c[:-1], prec_c[:-1]], delimiter=",",
                       header="threshold,recall,precision", comments="")

        tqdm.write(f"[{epoch:03d}/{epochs}] "
                   f"train_loss={tr_loss:.4f} acc={tr_acc:.3f} | "
                   f"val_loss={vl_loss:.4f} acc={vl_acc:.3f} | "
                   f"AUC={roc_auc:.3f} AP={pr_auc:.3f} F1={f1:.3f} | "
                   f"lr={lr_now:.2e} | {epoch_sec:.1f}s")

        # === 可选：向 Optuna 上报 + 剪枝（仍按 val_acc） ===
        if report_to_optuna is not None:
            report_to_optuna.report(vl_acc, step=epoch)
            if prune_on and report_to_optuna.should_prune():
                raise optuna.TrialPruned()

    # ===== 训练结束：保存 loss/acc 曲线 =====
    epochs_range = range(1, epochs + 1)
    tr_loss = [x[0] for x in train_hist]; tr_acc = [x[1] for x in train_hist]
    vl_loss = [x[0] for x in val_hist];   vl_acc = [x[1] for x in val_hist]
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1); plt.plot(epochs_range, tr_loss, label="train")
    plt.plot(epochs_range, vl_loss, label="val"); plt.title("Loss"); plt.grid(); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(epochs_range, tr_acc, label="train")
    plt.plot(epochs_range, vl_acc, label="val"); plt.title("Accuracy"); plt.grid(); plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "loss_acc_curve.png", dpi=200)
    print(f"\n✓ Logs & curves saved in: {run_dir}")

    return best_acc

# ===== 你原来的 main（保持运行逻辑不变） =====
# 原来是batch_size=256 , num_workers=4  , warmup_epochs=3 , lr=3e-4
def main():
    best_acc = run_training(
        epochs=80, warmup_epochs=3, lr=3e-4, weight_decay=1e-4,
        batch_size=256, num_workers=0, label_smoothing=0.05, dropout=0.0
    )
    print(f"Finished Training. Best val_acc={best_acc:.4f}")

# ===== Optuna 搜索入口（原逻辑） =====
if USE_OPTUNA:
    import optuna
    def objective(trial: "optuna.trial.Trial"):
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 192, 256, 320])
        warmup_epochs = trial.suggest_int("warmup_epochs", 0, 5)
        label_smooth = trial.suggest_float("label_smoothing", 0.0, 0.1)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)

        lite_tf_blocks = trial.suggest_int("lite_tf_blocks", 0, 2)  # 0 表示关闭
        lite_tf_dim = trial.suggest_categorical("lite_tf_dim", [128, 160, 192])
        heads_candidates = {128: [4, 8], 160: [5, 10], 192: [3, 6, 8]}
        lite_tf_heads = trial.suggest_categorical("lite_tf_heads", heads_candidates[lite_tf_dim])
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

        try:
            optuna_log_root = Path(os.environ.get(
                "OPTUNA_LOG_ROOT", str(CHECKPOINTS_DIR / "optuna" / "logs")))
            optuna_paper_root = Path(os.environ.get(
                "OPTUNA_PAPER_ROOT", str(CHECKPOINTS_DIR / "optuna" / "paper")))
            best_acc = run_training(
                epochs=TUNE_EPOCHS, warmup_epochs=warmup_epochs,
                lr=lr, weight_decay=weight_decay,
                batch_size=batch_size, num_workers=4,
                label_smoothing=label_smooth, dropout=dropout,
                model_kwargs=model_kwargs,
                log_root=str(optuna_log_root / f"trial_{trial.number}"),
                paper_root=str(optuna_paper_root / f"trial_{trial.number}"),
                report_to_optuna=trial, prune_on=PRUNE_ON
            )
        except optuna.TrialPruned:
            raise
        return best_acc

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=max(2, TUNE_EPOCHS//4))
    study = optuna.create_study(direction="maximize", pruner=pruner, study_name="convnext_tiny31_search")
    study.optimize(objective, n_trials=N_TRIALS)
    print("\n=== Optuna Best ===")
    print("best_value (val_acc):", study.best_value)
    print("best_params:", study.best_params)

# Windows 入口保护
if __name__ == "__main__" and not USE_OPTUNA:
    torch.multiprocessing.freeze_support()
    main()
