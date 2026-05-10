#!/usr/bin/env python3
"""
CICIoT2023 — Autoencoder-based Anomaly Detector
Architecture : Symmetric deep autoencoder (BN + GELU, residual skip in encoder/decoder)
Loss         : MSE reconstruction loss
Optimizer    : AdamW + OneCycleLR
Threshold    : calibrated at a chosen percentile of val-set reconstruction errors

Training data: ALL known classes (benign + attacks).
At inference time, reconstruction error > threshold → unknown / novel attack.
"""

import argparse
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
from sklearn.manifold import TSNE
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
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

# ─── Hyperparameters ─────────────────────────────────────────────────────────
MAX_SAMPLES_PER_CLASS = 50_000
LATENT_DIM            = 16      # bottleneck size; meaningful compression of 39 features
HIDDEN_DIMS           = [256, 128, 64]  # encoder layers (reversed for decoder)
DROPOUT               = 0.10    # lighter dropout — reconstruction task benefits from it
BATCH_SIZE            = 4096
EPOCHS                = 50
LR                    = 1e-3
WEIGHT_DECAY          = 1e-4
GRAD_CLIP             = 1.0
EARLY_STOP_PATIENCE   = 8
THRESHOLD_PERCENTILE  = 99.0   # calibrated on val set; tune for precision/recall trade-off

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
        logging.FileHandler(OUTPUT_DIR / "ae_train.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING  (same stratified loader as train.py)
# ═══════════════════════════════════════════════════════════════════════════════

def load_stratified(data_dir: Path, max_per_class: int) -> pd.DataFrame:
    """Same stratified loader as train.py — shares the same on-disk cache."""
    cache_path = data_dir / f".cache_all_{max_per_class}_s{SEED}.npz"

    if cache_path.exists():
        log.info(f"Loading from cache: {cache_path}")
        raw  = np.load(cache_path, allow_pickle=True)
        df   = pd.DataFrame(raw["X"], columns=FEATURE_COLS)
        df[LABEL_COL] = raw["labels"].astype(str)
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
            already    = sum(len(d) for d in buckets[label])
            remaining  = max_per_class - already
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
    def __init__(self):
        self.scaler = RobustScaler()
        self.le     = LabelEncoder()

    def fit_transform(self, df: pd.DataFrame):
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS)

        X = df[FEATURE_COLS].values.astype(np.float32)
        y = self.le.fit_transform(df[LABEL_COL].values)

        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=SEED
        )
        X_val, X_te, y_val, y_te = train_test_split(
            X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED
        )

        X_tr  = self.scaler.fit_transform(X_tr).astype(np.float32)
        X_val = self.scaler.transform(X_val).astype(np.float32)
        X_te  = self.scaler.transform(X_te).astype(np.float32)

        clip  = 20.0
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


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET  (unsupervised — only features, no labels)
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureDataset(Dataset):
    """Labels are stored only for threshold analysis; training ignores them."""
    def __init__(self, X: np.ndarray, y: np.ndarray | None = None):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def make_loaders(X_tr, X_val, X_te, y_tr, y_val, y_te):
    nw = min(4, os.cpu_count() or 1)
    kw = dict(pin_memory=USE_AMP, num_workers=nw, persistent_workers=(nw > 0))

    train_loader = DataLoader(
        FeatureDataset(X_tr),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True, **kw,
    )
    val_loader = DataLoader(
        FeatureDataset(X_val, y_val),
        batch_size=BATCH_SIZE * 2, shuffle=False, **kw,
    )
    test_loader = DataLoader(
        FeatureDataset(X_te, y_te),
        batch_size=BATCH_SIZE * 2, shuffle=False, **kw,
    )
    return train_loader, val_loader, test_loader


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL — Deep Symmetric Autoencoder
# ═══════════════════════════════════════════════════════════════════════════════

class AEBlock(nn.Module):
    """
    One encoder/decoder layer: Linear → BN → GELU → Dropout.
    A residual skip is added when in_dim == out_dim (used inside the bottleneck
    transition layers to preserve gradient flow through the deep stack).
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.fc   = nn.Linear(in_dim, out_dim, bias=False)
        self.bn   = nn.BatchNorm1d(out_dim)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.bn(self.fc(x)))) + self.skip(x)


class Autoencoder(nn.Module):
    """
    Symmetric autoencoder for tabular network-flow anomaly detection.

    Architecture (default HIDDEN_DIMS=[256,128,64], LATENT_DIM=16)
    ─────────────────────────────────────────────────────────────────
    Encoder:  39 → 256 → 128 → 64 → 16  (bottleneck)
    Decoder:  16 → 64  → 128 → 256 → 39

    Each non-terminal step is an AEBlock (Linear + BN + GELU + Dropout + skip).
    The final decoder output is a plain linear layer with no activation so that
    the reconstruction lives in the same unbounded space as the scaled input.
    """

    def __init__(self, in_dim: int, hidden_dims: list[int],
                 latent_dim: int, dropout: float):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        enc_layers = []
        prev = in_dim
        for h in hidden_dims:
            enc_layers.append(AEBlock(prev, h, dropout))
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim, bias=False))
        enc_layers.append(nn.BatchNorm1d(latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # ── Decoder ──────────────────────────────────────────────────────────
        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.append(AEBlock(prev, h, dropout))
            prev = h
        dec_layers.append(nn.Linear(prev, in_dim))   # no BN/activation at output
        self.decoder = nn.Sequential(*dec_layers)

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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE between input and reconstruction. Shape: (B,)"""
        x_hat = self.forward(x)
        return F.mse_loss(x_hat, x, reduction="none").mean(dim=1)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_epoch(model, loader, optimizer=None, scaler_amp=None, scheduler=None):
    """Unified train/eval pass. optimizer=None → eval mode."""
    training = optimizer is not None
    model.train(training)

    total_loss = n = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            # FeatureDataset returns just X during training (no labels)
            X = batch[0] if isinstance(batch, (list, tuple)) else batch
            X = X.to(DEVICE, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=USE_AMP):
                X_hat = model(X)

            loss = F.mse_loss(X_hat.float(), X.float())

            if training:
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler_amp.step(optimizer)
                scaler_amp.update()
                scheduler.step()

            total_loss += loss.item() * len(X)
            n          += len(X)

    return total_loss / n


# ═══════════════════════════════════════════════════════════════════════════════
#  THRESHOLD CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_errors(model, loader) -> tuple[np.ndarray, np.ndarray]:
    """Return (per-sample reconstruction errors, class labels) from a labelled loader."""
    model.eval()
    errors, labels = [], []
    for X, y in loader:
        X = X.to(DEVICE, non_blocking=True)
        err = model.reconstruction_error(X).cpu().numpy()
        errors.append(err)
        labels.append(y.numpy())
    return np.concatenate(errors), np.concatenate(labels)


def calibrate_threshold(errors: np.ndarray, percentile: float) -> float:
    threshold = float(np.percentile(errors, percentile))
    log.info(
        f"Threshold @ {percentile}th percentile: {threshold:.6f}  "
        f"(val error — mean={errors.mean():.4f}, std={errors.std():.4f}, "
        f"max={errors.max():.4f})"
    )
    return threshold


# ═══════════════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(history: dict, out_dir: Path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history["train_loss"], label="Train MSE", linewidth=1.8)
    ax.plot(epochs, history["val_loss"],   label="Val MSE",   linewidth=1.8)
    ax.set_title("Autoencoder — Reconstruction Loss (MSE)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "ae_training_curves.png", dpi=150)
    plt.close()
    log.info("Saved outputs/ae_training_curves.png")


def plot_error_distribution(errors: np.ndarray, labels: np.ndarray,
                             classes: np.ndarray, threshold: float,
                             out_dir: Path, tag: str = "val"):
    """
    KDE plot of reconstruction error per class + threshold line.
    Helps verify that the threshold separates known traffic from the tail.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot overall distribution
    ax.hist(errors, bins=200, density=True, alpha=0.3, color="steelblue", label="All")

    # Overlay per-class medians as vertical tick marks
    class_medians = {}
    for cls_idx in np.unique(labels):
        mask = labels == cls_idx
        class_medians[classes[cls_idx]] = np.median(errors[mask])

    ax.axvline(threshold, color="red", linewidth=2,
               linestyle="--", label=f"Threshold ({THRESHOLD_PERCENTILE}th pct) = {threshold:.4f}")

    ax.set_xlabel("Reconstruction Error (MSE per sample)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"Autoencoder Reconstruction Error Distribution — {tag} set",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / f"ae_error_dist_{tag}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"Saved {out_path}")


@torch.no_grad()
def collect_latents(model, loader) -> tuple[np.ndarray, np.ndarray]:
    """Return (latent vectors, integer labels) from a labelled loader."""
    model.eval()
    zs, ys = [], []
    for X, y in loader:
        X = X.to(DEVICE, non_blocking=True)
        zs.append(model.encode(X).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(zs), np.concatenate(ys)


def plot_latent_tsne(latents: np.ndarray, labels: np.ndarray,
                     classes: np.ndarray, out_dir: Path):
    N   = min(6000, len(latents))
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(latents), N, replace=False)
    log.info(f"Running t-SNE on {N} latent vectors …")
    emb = TSNE(n_components=2, random_state=SEED, perplexity=40,
               max_iter=500, init="pca").fit_transform(latents[idx])

    fig, ax = plt.subplots(figsize=(13, 9))
    cmap = plt.get_cmap("tab20", len(classes))
    for i, cls in enumerate(classes):
        mask = labels[idx] == i
        if mask.sum() == 0:
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1], s=7, alpha=0.5,
                   color=cmap(i), label=cls, linewidths=0)
    ax.set_title("Autoencoder — t-SNE of Latent Space (val set)",
                 fontsize=13, fontweight="bold")
    ax.legend(markerscale=3, fontsize=7, ncol=2, loc="best",
              framealpha=0.7, edgecolor="none")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_dir / "ae_latent_tsne.png", dpi=150)
    plt.close()
    log.info("Saved outputs/ae_latent_tsne.png")


def plot_per_class_errors(errors: np.ndarray, labels: np.ndarray,
                           classes: np.ndarray, threshold: float, out_dir: Path):
    """Box-plot of reconstruction error per class with threshold overlay."""
    class_errors = {classes[i]: errors[labels == i] for i in np.unique(labels)}
    sorted_classes = sorted(class_errors, key=lambda c: np.median(class_errors[c]))
    medians = [np.median(class_errors[c]) for c in sorted_classes]

    fig, ax = plt.subplots(figsize=(14, max(8, len(sorted_classes) * 0.4)))
    ax.boxplot(
        [class_errors[c] for c in sorted_classes],
        vert=False, labels=sorted_classes,
        flierprops=dict(marker=".", markersize=2, alpha=0.3),
        patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.6),
    )
    ax.axvline(threshold, color="red", linewidth=2, linestyle="--",
               label=f"Threshold = {threshold:.4f}")
    ax.set_xlabel("Reconstruction Error (MSE)", fontsize=11)
    ax.set_title("Per-Class Reconstruction Error — Validation Set",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "ae_per_class_errors.png", dpi=150)
    plt.close()
    log.info("Saved outputs/ae_per_class_errors.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def _restore_scaler(center: np.ndarray, scale: np.ndarray) -> RobustScaler:
    """Rebuild a RobustScaler from saved center/scale without refitting."""
    scaler = RobustScaler()
    scaler.center_ = center
    scaler.scale_  = scale
    scaler.n_features_in_ = len(center)
    return scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", metavar="CKPT",
        help="Path to ae_model.pt — skip training and go straight to t-SNE plot.",
    )
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("CICIoT2023 — Autoencoder Anomaly Detector Training")
    log.info(f"Device  : {DEVICE}  |  AMP: {USE_AMP}")
    log.info(f"PyTorch : {torch.__version__}")
    if DEVICE.type == "cuda":
        log.info(f"GPU     : {torch.cuda.get_device_name(0)}")
    log.info("=" * 70)

    # ── 1. Load & preprocess ──────────────────────────────────────────────────
    df = load_stratified(DATA_DIR, MAX_SAMPLES_PER_CLASS)

    if args.resume:
        # Restore scaler from checkpoint so the split uses identical scaling
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        log.info(f"Loaded checkpoint from {args.resume}")
        scaler  = _restore_scaler(ckpt["scaler_mean"], ckpt["scaler_scale"])
        classes = ckpt["classes"]
        cfg     = ckpt["config"]

        # Recreate LabelEncoder with saved class order
        le = LabelEncoder()
        le.classes_ = classes

        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS)
        X  = df[FEATURE_COLS].values.astype(np.float32)
        y  = le.transform(df[LABEL_COL].values)

        _, X_tmp, _, y_tmp = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=SEED
        )
        X_val, X_te, y_val, y_te = train_test_split(
            X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED
        )
        clip  = 20.0
        X_val = np.clip(scaler.transform(X_val).astype(np.float32), -clip, clip)
        X_te  = np.clip(scaler.transform(X_te).astype(np.float32),  -clip, clip)
        del df

        n_features = X_val.shape[1]
        nw  = min(4, os.cpu_count() or 1)
        kw  = dict(pin_memory=USE_AMP, num_workers=nw, persistent_workers=(nw > 0))
        val_loader  = DataLoader(FeatureDataset(X_val, y_val),
                                 batch_size=BATCH_SIZE * 2, shuffle=False, **kw)

        model = Autoencoder(
            in_dim=n_features,
            hidden_dims=cfg["hidden_dims"],
            latent_dim=cfg["latent_dim"],
            dropout=cfg["dropout"],
        ).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        log.info("Model weights restored — skipping training.")

        log.info("Generating latent space visualisation …")
        latents, lat_labels = collect_latents(model, val_loader)
        plot_latent_tsne(latents, lat_labels, classes, OUTPUT_DIR)

        log.info(f"\nAll outputs written to: {OUTPUT_DIR.resolve()}")
        log.info("Done.")
        return

    # ── Full training path ────────────────────────────────────────────────────
    prep = Preprocessor()
    X_tr, X_val, X_te, y_tr, y_val, y_te = prep.fit_transform(df)
    classes = prep.classes_
    del df

    n_features = X_tr.shape[1]
    log.info(f"Features: {n_features}  |  Classes: {len(classes)}")

    # ── 2. DataLoaders ────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = make_loaders(
        X_tr, X_val, X_te, y_tr, y_val, y_te
    )

    # ── 3. Model ──────────────────────────────────────────────────────────────
    model = Autoencoder(
        in_dim=n_features,
        hidden_dims=HIDDEN_DIMS,
        latent_dim=LATENT_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)
    log.info(f"Autoencoder params: {model.count_params():,}")
    log.info(f"Architecture: {n_features} → {HIDDEN_DIMS} → {LATENT_DIM} → "
             f"{list(reversed(HIDDEN_DIMS))} → {n_features}")

    # ── 4. Optimiser & scheduler ──────────────────────────────────────────────
    optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = EPOCHS * len(train_loader)
    scheduler   = OneCycleLR(
        optimizer, max_lr=LR,
        total_steps=total_steps,
        pct_start=0.08,
        anneal_strategy="cos",
        final_div_factor=1e3,
    )
    amp_scaler = GradScaler("cuda", enabled=USE_AMP)

    # ── 5. Training loop ──────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state    = None
    patience_cnt  = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        tr_loss = run_epoch(model, train_loader, optimizer, amp_scaler, scheduler)
        vl_loss = run_epoch(model, val_loader)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)

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
            f"tr_mse={tr_loss:.6f}  vl_mse={vl_loss:.6f}  "
            f"{time.time() - t0:.1f}s{flag}"
        )

        if patience_cnt >= EARLY_STOP_PATIENCE:
            log.info(f"Early stopping triggered at epoch {epoch}.")
            break

    # ── 6. Restore best weights ───────────────────────────────────────────────
    if best_state:
        model.load_state_dict(best_state)

    # ── 7. Calibrate threshold on validation set ──────────────────────────────
    log.info("\nCalibrating anomaly threshold on validation set …")
    val_errors, val_labels = compute_errors(model, val_loader)
    threshold = calibrate_threshold(val_errors, THRESHOLD_PERCENTILE)

    flagged_pct = 100.0 * (val_errors > threshold).mean()
    log.info(f"Val set flagged as anomalous: {flagged_pct:.2f}%  "
             f"(expected ~{100 - THRESHOLD_PERCENTILE:.1f}%)")

    # ── 8. Test-set error stats ───────────────────────────────────────────────
    log.info("\nComputing test-set reconstruction errors …")
    te_errors, te_labels = compute_errors(model, test_loader)
    te_flagged = 100.0 * (te_errors > threshold).mean()
    log.info(f"Test set mean error  : {te_errors.mean():.6f}")
    log.info(f"Test set flagged     : {te_flagged:.2f}%")

    # ── 9. Save checkpoint ────────────────────────────────────────────────────
    ckpt_path = OUTPUT_DIR / "ae_model.pt"
    torch.save(
        {
            "model_state": best_state,
            "threshold":   threshold,
            "classes":     classes,
            "config": {
                "in_dim":      n_features,
                "hidden_dims": HIDDEN_DIMS,
                "latent_dim":  LATENT_DIM,
                "dropout":     DROPOUT,
            },
            "scaler_mean":  prep.scaler.center_,
            "scaler_scale": prep.scaler.scale_,
        },
        ckpt_path,
    )
    log.info(f"Checkpoint saved → {ckpt_path}")

    # ── 10. Plots ─────────────────────────────────────────────────────────────
    plot_training_curves(history, OUTPUT_DIR)
    plot_error_distribution(val_errors, val_labels, classes, threshold, OUTPUT_DIR, tag="val")
    plot_error_distribution(te_errors,  te_labels,  classes, threshold, OUTPUT_DIR, tag="test")
    plot_per_class_errors(val_errors, val_labels, classes, threshold, OUTPUT_DIR)

    log.info("Generating latent space visualisation …")
    latents, lat_labels = collect_latents(model, val_loader)
    plot_latent_tsne(latents, lat_labels, classes, OUTPUT_DIR)

    log.info(f"\nAll outputs written to: {OUTPUT_DIR.resolve()}")
    log.info("Done.")


if __name__ == "__main__":
    main()
