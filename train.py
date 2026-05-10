#!/usr/bin/env python3
"""
CICIoT2023 — IoT Network Intrusion Detection
Architecture : Deep Residual MLP (pre-activation residual blocks)
Loss         : Multi-class Focal Loss + label smoothing
Optimizer    : AdamW + OneCycleLR
Handles 34-class severe class imbalance via focal loss + inverse-frequency weights
"""

import os
import sys
import time
import random
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, balanced_accuracy_score,
)
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import GradScaler, autocast

warnings.filterwarnings("ignore")

# ─── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "MERGED_CSV"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Hardware ────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"  # mixed-precision only on GPU

# ─── Hyperparameters ─────────────────────────────────────────────────────────
MAX_SAMPLES_PER_CLASS = 50_000   # stratified cap keeps training feasible on 4 GB VRAM
HIDDEN_DIM            = 512
N_BLOCKS              = 6        # residual blocks
DROPOUT               = 0.30
BATCH_SIZE            = 4096     # large batch plays well with BN + focal loss
EPOCHS                = 50
LR                    = 3e-4
WEIGHT_DECAY          = 1e-4
LABEL_SMOOTHING       = 0.05
GRAD_CLIP             = 1.0
FOCAL_GAMMA           = 2.0      # focuses loss on hard/minority examples
EARLY_STOP_PATIENCE   = 8

# ─── Feature / Label names ───────────────────────────────────────────────────
FEATURE_COLS = [
    "Header_Length", "Protocol Type", "Time_To_Live", "Rate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number",
    "psh_flag_number", "ack_flag_number", "ece_flag_number",
    "cwr_flag_number", "ack_count", "syn_count", "fin_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC",
    "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size",
    "IAT", "Number", "Variance",
]
LABEL_COL = "Label"

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "train.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING  (stratified sample across all 63 CSV files)
# ═══════════════════════════════════════════════════════════════════════════════

def load_stratified(data_dir: Path, max_per_class: int) -> pd.DataFrame:
    """
    Stream every CSV; collect up to max_per_class rows per label.
    Result is cached as a compressed .npz so subsequent runs load in seconds.
    Delete the cache file to force a fresh scan.
    """
    cache_path = data_dir / f".cache_all_{max_per_class}_s{SEED}.npz"

    if cache_path.exists():
        log.info(f"Loading from cache: {cache_path}")
        raw   = np.load(cache_path, allow_pickle=True)
        X     = raw["X"]
        lbls  = raw["labels"].astype(str)
        df    = pd.DataFrame(X, columns=FEATURE_COLS)
        df[LABEL_COL] = lbls
        log.info(f"Cache loaded: {len(df):,} rows, {df[LABEL_COL].nunique()} classes")
        return df

    csv_files = sorted(data_dir.glob("*.csv"))
    log.info(f"Found {len(csv_files)} CSV files in {data_dir} — building cache …")

    buckets: dict[str, list[pd.DataFrame]] = {}

    for fpath in tqdm(csv_files, desc="Loading CSVs", unit="file"):
        try:
            chunk = pd.read_csv(fpath, low_memory=False)
        except Exception as exc:
            log.warning(f"Skipping {fpath.name}: {exc}")
            continue

        needed_cols = FEATURE_COLS + [LABEL_COL]
        missing = [c for c in needed_cols if c not in chunk.columns]
        if missing:
            log.warning(f"{fpath.name}: missing {missing} — skipped")
            continue

        chunk = chunk[needed_cols]

        for label, grp in chunk.groupby(LABEL_COL, sort=False):
            if label not in buckets:
                buckets[label] = []
            already = sum(len(d) for d in buckets[label])
            remaining = max_per_class - already
            if remaining <= 0:
                continue
            buckets[label].append(grp.iloc[:remaining])

    log.info(f"\nStratified sample — {len(buckets)} unique classes:")
    parts = []
    for label in sorted(buckets):
        df_cls = pd.concat(buckets[label], ignore_index=True)
        log.info(f"  {label:<45s} {len(df_cls):>7,}")
        parts.append(df_cls)

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    log.info(f"\nTotal rows loaded: {len(combined):,}")

    np.savez_compressed(
        cache_path,
        X=combined[FEATURE_COLS].values.astype(np.float32),
        labels=combined[LABEL_COL].values.astype(str),
    )
    log.info(f"Cache saved → {cache_path}")
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

class Preprocessor:
    """Fits RobustScaler + LabelEncoder on training data; transforms all splits."""

    def __init__(self):
        self.scaler = RobustScaler()
        self.le     = LabelEncoder()

    def fit_transform(self, df: pd.DataFrame):
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS)

        X = df[FEATURE_COLS].values.astype(np.float32)
        y = self.le.fit_transform(df[LABEL_COL].values).astype(np.int64)

        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=SEED
        )
        X_val, X_te, y_val, y_te = train_test_split(
            X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED
        )

        # RobustScaler: trims outliers via IQR — ideal for network traffic metrics.
        # Clip to [-20, 20] after scaling: features like IAT have IQR ≈ 6e-5 s,
        # so even moderate outliers scale to 10^7+, which overflows float16 in AMP.
        X_tr  = self.scaler.fit_transform(X_tr).astype(np.float32)
        X_val = self.scaler.transform(X_val).astype(np.float32)
        X_te  = self.scaler.transform(X_te).astype(np.float32)
        clip = 20.0
        X_tr  = np.clip(X_tr,  -clip, clip)
        X_val = np.clip(X_val, -clip, clip)
        X_te  = np.clip(X_te,  -clip, clip)

        log.info(
            f"Split sizes — train: {len(X_tr):,}  val: {len(X_val):,}  test: {len(X_te):,}"
        )
        return X_tr, X_val, X_te, y_tr, y_val, y_te

    @property
    def classes_(self):
        return self.le.classes_

    @property
    def n_classes(self):
        return len(self.le.classes_)


def compute_class_weights(y_train: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights clipped at 10× to prevent extreme gradients."""
    counts  = np.bincount(y_train, minlength=n_classes).astype(float)
    weights = counts.sum() / (n_classes * np.maximum(counts, 1))
    weights = np.clip(weights, None, 10.0)
    return torch.tensor(weights, dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class FlowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loaders(X_tr, X_val, X_te, y_tr, y_val, y_te):
    nw = min(4, os.cpu_count() or 1)
    kw = dict(pin_memory=USE_AMP, num_workers=nw, persistent_workers=(nw > 0))

    train_loader = DataLoader(
        FlowDataset(X_tr, y_tr),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True, **kw,
    )
    val_loader = DataLoader(
        FlowDataset(X_val, y_val),
        batch_size=BATCH_SIZE * 2, shuffle=False, **kw,
    )
    test_loader = DataLoader(
        FlowDataset(X_te, y_te),
        batch_size=BATCH_SIZE * 2, shuffle=False, **kw,
    )
    return train_loader, val_loader, test_loader


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL  — Deep Residual MLP
# ═══════════════════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    """
    Pre-activation residual block.
    BN → GELU → Linear → BN → GELU → Dropout → Linear  +  skip

    Pre-activation (He et al. 2016) places BN/activation before the weight
    layer, which improves gradient flow and avoids the "dying ReLU" issue
    when stacking many blocks on tabular data.
    """

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResidualMLP(nn.Module):
    """
    Deep Residual MLP for tabular network-flow classification.

    Architecture
    ─────────────────────────────────────────────────────────────
    Input (39)
      └→ Stem: Linear(39→512) + BN + GELU + Dropout(0.15)
           └→ ResBlock × 6  (512-dim throughout)
                └→ Head: BN → Linear(512→256) → GELU → Dropout → Linear(256→34)

    Design rationale
    ─────────────────────────────────────────────────────────────
    - Residual connections: essential for >4 layers to prevent gradient vanishing
    - BatchNorm: stabilises training over the wide range of network-traffic scales
    - GELU: smooth non-linearity; better than ReLU for deep tabular networks
    - Wide-then-narrow head: compresses learned representation before final softmax
    """

    def __init__(self, in_dim: int, n_classes: int,
                 hidden: int, n_blocks: int, dropout: float):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        self.tower = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(n_blocks)])

        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden // 2, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.tower(self.stem(x)))

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════════
#  LOSS  — Multi-class Focal Loss with label smoothing
# ═══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Multi-class focal loss (Lin et al. 2017).

    γ (gamma) down-weights easy examples so the model focuses on
    hard-to-classify minority attack classes.  Combined with inverse-frequency
    class weights and label smoothing for maximum robustness to imbalance.
    """

    def __init__(self, weight: torch.Tensor, gamma: float, label_smoothing: float):
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cast to float32 — AMP delivers float16 logits; softmax/log chains
        # produce NaN/-Inf in float16 for near-zero probabilities.
        logits = logits.float()
        C = logits.size(1)

        log_p = F.log_softmax(logits, dim=1)           # (B, C) — numerically stable

        with torch.no_grad():
            # Focal weight: down-weight easy examples
            p_t   = log_p.exp().gather(1, targets.unsqueeze(1)).squeeze(1)
            p_t   = p_t.clamp(min=1e-7, max=1.0 - 1e-7)
            focal = (1.0 - p_t) ** self.gamma          # (B,)
            cls_w = self.weight.to(logits.device)[targets]  # (B,)

            # Label-smoothed target distribution
            smooth = torch.full_like(logits, self.label_smoothing / (C - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        ce   = -(smooth * log_p).sum(1)                # (B,)
        return (focal * cls_w * ce).mean()


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_epoch(model, loader, criterion, optimizer=None, scaler_amp=None, scheduler=None):
    """Unified train / eval pass.  Pass optimizer=None for eval."""
    training = optimizer is not None
    model.train(training)

    total_loss = correct = n = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for X, y in loader:
            X, y = X.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=USE_AMP):
                logits = model(X)
            # Loss outside autocast: FocalLoss casts to float32 internally;
            # this ensures the GradScaler scales a valid float32 loss.
            loss = criterion(logits, y)

            if training:
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler_amp.step(optimizer)
                scaler_amp.update()
                scheduler.step()

            total_loss += loss.item() * len(y)
            correct    += (logits.detach().float().argmax(1) == y).sum().item()
            n          += len(y)

    return total_loss / n, correct / n


@torch.no_grad()
def collect_predictions(model, loader):
    model.eval()
    preds, trues = [], []
    for X, y in loader:
        X = X.to(DEVICE, non_blocking=True)
        preds.append(model(X).argmax(1).cpu().numpy())
        trues.append(y.numpy())
    return np.concatenate(preds), np.concatenate(trues)


# ═══════════════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(history: dict, out_dir: Path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (tr_key, vl_key), title in zip(
        axes,
        [("train_loss", "val_loss"), ("train_acc", "val_acc")],
        ["Loss", "Accuracy"],
    ):
        ax.plot(epochs, history[tr_key], label="Train", linewidth=1.8)
        ax.plot(epochs, history[vl_key], label="Val",   linewidth=1.8)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("CICIoT2023 — Training Progress", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close()
    log.info("Saved outputs/training_curves.png")


def plot_confusion_matrix(y_true, y_pred, classes, out_dir: Path):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(22, 20))
    sns.heatmap(
        cm_norm, annot=False, cmap="Blues",
        xticklabels=classes, yticklabels=classes, ax=ax,
        linewidths=0.4, linecolor="lightgray", vmin=0, vmax=1,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_title("Normalised Confusion Matrix — Test Set", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0,  fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    log.info("Saved outputs/confusion_matrix.png")


def plot_per_class_f1(report: dict, classes, out_dir: Path):
    f1s    = [report.get(c, {}).get("f1-score", 0.0) for c in classes]
    colors = plt.cm.RdYlGn([v for v in f1s])

    fig, ax = plt.subplots(figsize=(12, max(8, len(classes) * 0.35)))
    bars = ax.barh(classes, f1s, color=colors, edgecolor="white", height=0.7)
    for bar, val in zip(bars, f1s):
        ax.text(
            min(val + 0.01, 0.99), bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=8,
        )
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("F1-Score", fontsize=11)
    ax.set_title("Per-Class F1-Score — Test Set", fontsize=13, fontweight="bold")
    ax.axvline(0.9, color="green",  linestyle="--", alpha=0.5, linewidth=1, label="F1=0.90")
    ax.axvline(0.7, color="orange", linestyle="--", alpha=0.5, linewidth=1, label="F1=0.70")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "per_class_f1.png", dpi=150)
    plt.close()
    log.info("Saved outputs/per_class_f1.png")


def plot_class_distribution(y_train, classes, out_dir: Path):
    counts = np.bincount(y_train, minlength=len(classes))
    idx    = np.argsort(counts)[::-1]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(range(len(classes)), counts[idx], color="steelblue", edgecolor="white")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([classes[i] for i in idx], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Sample count")
    ax.set_title("Training Set Class Distribution (after stratified cap)", fontsize=13)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "class_distribution.png", dpi=150)
    plt.close()
    log.info("Saved outputs/class_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 70)
    log.info("CICIoT2023 — IoT Intrusion Detection (Deep Residual MLP)")
    log.info(f"Device  : {DEVICE}  |  AMP: {USE_AMP}")
    log.info(f"PyTorch : {torch.__version__}")
    if DEVICE.type == "cuda":
        log.info(f"GPU     : {torch.cuda.get_device_name(0)}")
    log.info("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    df = load_stratified(DATA_DIR, MAX_SAMPLES_PER_CLASS)

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    prep = Preprocessor()
    X_tr, X_val, X_te, y_tr, y_val, y_te = prep.fit_transform(df)
    del df  # free memory

    n_features = X_tr.shape[1]
    n_classes  = prep.n_classes
    classes    = prep.classes_
    log.info(f"Features: {n_features}  |  Classes: {n_classes}")

    plot_class_distribution(y_tr, classes, OUTPUT_DIR)

    # ── 3. DataLoaders ────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = make_loaders(
        X_tr, X_val, X_te, y_tr, y_val, y_te
    )

    # ── 4. Model ──────────────────────────────────────────────────────────────
    model = ResidualMLP(
        in_dim=n_features, n_classes=n_classes,
        hidden=HIDDEN_DIM, n_blocks=N_BLOCKS, dropout=DROPOUT,
    ).to(DEVICE)
    log.info(f"Model params: {model.count_params():,}")

    # ── 5. Loss, optimiser, scheduler ─────────────────────────────────────────
    class_weights = compute_class_weights(y_tr, n_classes).to(DEVICE)
    criterion = FocalLoss(
        weight=class_weights,
        gamma=FOCAL_GAMMA,
        label_smoothing=LABEL_SMOOTHING,
    )
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = EPOCHS * len(train_loader)
    scheduler = OneCycleLR(
        optimizer, max_lr=LR,
        total_steps=total_steps,
        pct_start=0.08,          # 8 % warmup
        anneal_strategy="cos",
        final_div_factor=1e3,
    )
    amp_scaler = GradScaler("cuda", enabled=USE_AMP)

    # ── 6. Training loop ──────────────────────────────────────────────────────
    history = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc")}
    best_val_loss  = float("inf")
    best_f1_macro  = 0.0
    best_state     = None
    patience_cnt   = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        tr_loss, tr_acc = run_epoch(
            model, train_loader, criterion, optimizer, amp_scaler, scheduler
        )
        vl_loss, vl_acc = run_epoch(model, val_loader, criterion)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        improved = vl_loss < best_val_loss
        if improved:
            best_val_loss = vl_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt  = 0
            flag = "  ✓ best"
        else:
            patience_cnt += 1
            flag = f"  (patience {patience_cnt}/{EARLY_STOP_PATIENCE})"

        log.info(
            f"Epoch {epoch:3d}/{EPOCHS}  "
            f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}  "
            f"vl_loss={vl_loss:.4f}  vl_acc={vl_acc:.4f}  "
            f"{time.time() - t0:.1f}s{flag}"
        )

        if patience_cnt >= EARLY_STOP_PATIENCE:
            log.info(f"Early stopping triggered at epoch {epoch}.")
            break

    # ── 7. Restore best weights & save checkpoint ─────────────────────────────
    if best_state:
        model.load_state_dict(best_state)

    torch.save(
        {
            "model_state": best_state,
            "classes":     classes,
            "config": {
                "in_dim":    n_features,
                "n_classes": n_classes,
                "hidden":    HIDDEN_DIM,
                "n_blocks":  N_BLOCKS,
                "dropout":   DROPOUT,
            },
        },
        OUTPUT_DIR / "best_model.pt",
    )
    log.info(f"Model checkpoint saved → outputs/best_model.pt")

    # ── 8. Test-set evaluation ────────────────────────────────────────────────
    log.info("\nRunning test-set inference …")
    y_pred, y_true = collect_predictions(model, test_loader)

    acc     = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_mac  = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_wt   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    divider = "=" * 70
    log.info(f"\n{divider}")
    log.info("TEST SET RESULTS")
    log.info(f"  Accuracy           : {acc:.4f}   ({acc*100:.2f} %)")
    log.info(f"  Balanced Accuracy  : {bal_acc:.4f}   ({bal_acc*100:.2f} %)")
    log.info(f"  F1 Macro           : {f1_mac:.4f}")
    log.info(f"  F1 Weighted        : {f1_wt:.4f}")
    log.info(divider)

    report_dict = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True, zero_division=0
    )
    report_str = classification_report(
        y_true, y_pred, target_names=classes, zero_division=0
    )
    log.info(f"\nPer-class breakdown:\n{report_str}")

    summary_path = OUTPUT_DIR / "classification_report.txt"
    with open(summary_path, "w") as f:
        f.write(f"Accuracy           : {acc:.4f}\n")
        f.write(f"Balanced Accuracy  : {bal_acc:.4f}\n")
        f.write(f"F1 Macro           : {f1_mac:.4f}\n")
        f.write(f"F1 Weighted        : {f1_wt:.4f}\n\n")
        f.write(report_str)
    log.info(f"Report saved → {summary_path}")

    # ── 9. Plots ──────────────────────────────────────────────────────────────
    plot_training_curves(history, OUTPUT_DIR)
    plot_confusion_matrix(y_true, y_pred, classes, OUTPUT_DIR)
    plot_per_class_f1(report_dict, classes, OUTPUT_DIR)

    log.info(f"\nAll outputs written to: {OUTPUT_DIR.resolve()}")
    log.info("Done.")


if __name__ == "__main__":
    main()
