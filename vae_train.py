#!/usr/bin/env python3
"""
CICIoT2023 — Conditional Variational Autoencoder (CVAE)
Generates synthetic attack-type network flows for stress testing.

Architecture:
  Encoder : [x (39) || class_embed (64)] → ResBlocks → mu, log_var  (latent=64)
  Decoder : [z (64) || class_embed (64)] → ResBlocks → x_hat (39)

Training:
  Loss     = MSE reconstruction + β·KL divergence
  β anneals from 0 → 1 over first ANNEAL_EPOCHS (prevents posterior collapse)
  Trained only on attack classes (benign excluded) to specialise latent space.
"""

import os
import sys
import time
import random
import warnings
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

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

# ─── Feature / Label columns (must match train.py) ───────────────────────────
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
LABEL_COL   = "Label"
BENIGN_LABEL = "BenignTraffic"

# ─── Hyperparameters ─────────────────────────────────────────────────────────
MAX_SAMPLES_PER_CLASS = 50_000
LATENT_DIM            = 64
EMBED_DIM             = 64
HIDDEN_DIM            = 256
N_BLOCKS              = 4
DROPOUT               = 0.20
BATCH_SIZE            = 2048
EPOCHS                = 80
ANNEAL_EPOCHS         = 25      # β ramps 0→1 over this many epochs
LR                    = 1e-3
WEIGHT_DECAY          = 1e-5
EARLY_STOP_PATIENCE   = 12
CLIP_VAL              = 20.0    # matches train.py scaler clip

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "vae_train.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_attack_data(data_dir: Path, max_per_class: int):
    """Load stratified sample as numpy arrays, never materialising the full DataFrame."""
    csv_files = sorted(data_dir.glob("*.csv"))
    log.info(f"Found {len(csv_files)} CSV files")

    CHUNK_SIZE = 50_000
    needed     = FEATURE_COLS + [LABEL_COL]

    # Store raw float32 arrays per label; track count separately (O(1) lookup).
    X_buckets: dict[str, list[np.ndarray]] = {}
    counts:    dict[str, int]              = {}

    for fpath in tqdm(csv_files, desc="Loading CSVs", unit="file"):
        # Skip file entirely if every known class is already full.
        if counts and all(counts.get(lbl, 0) >= max_per_class for lbl in counts):
            log.info("All classes capped — skipping remaining files")
            break

        try:
            reader = pd.read_csv(
                fpath, usecols=needed, low_memory=False, chunksize=CHUNK_SIZE
            )
        except Exception as exc:
            log.warning(f"Skipping {fpath.name}: {exc}")
            continue

        file_skipped = False
        for chunk in reader:
            missing = [c for c in needed if c not in chunk.columns]
            if missing:
                log.warning(f"{fpath.name}: missing {missing} — skipped")
                file_skipped = True
                del chunk
                break

            for label, grp in chunk.groupby(LABEL_COL, sort=False):
                if label == BENIGN_LABEL:
                    continue
                already   = counts.get(label, 0)
                remaining = max_per_class - already
                if remaining <= 0:
                    continue
                rows = grp[FEATURE_COLS].values[:remaining].astype(np.float32)
                X_buckets.setdefault(label, []).append(rows)
                counts[label] = already + len(rows)

            del chunk   # release immediately — don't wait for GC

        if file_skipped:
            continue

    log.info(f"Attack classes found: {len(X_buckets)}")

    # Build final arrays one class at a time, freeing bucket lists as we go.
    label_names = sorted(X_buckets.keys())
    X_parts, y_parts = [], []
    for idx, label in enumerate(label_names):
        X_cls = np.concatenate(X_buckets.pop(label), axis=0)
        log.info(f"  {label:<45s} {len(X_cls):>7,}")
        X_parts.append(X_cls)
        y_parts.append(np.full(len(X_cls), idx, dtype=np.int64))

    X = np.concatenate(X_parts); del X_parts
    y = np.concatenate(y_parts); del y_parts

    perm = np.random.RandomState(SEED).permutation(len(X))
    log.info(f"Total attack rows: {len(X):,}")
    return X[perm], y[perm], np.array(label_names)


class AttackDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    """Pre-activation residual block (BN → GELU → Linear → BN → GELU → Drop → Linear)."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim, bias=False),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int,
                 embed_dim: int, latent_dim: int, n_blocks: int, dropout: float):
        super().__init__()
        self.label_embed = nn.Embedding(n_classes, embed_dim)
        self.stem = nn.Sequential(
            nn.Linear(in_dim + embed_dim, hidden, bias=False),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.tower  = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.mu     = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

    def forward(self, x, y):
        e   = self.label_embed(y)           # (B, embed_dim)
        h   = self.stem(torch.cat([x, e], dim=1))
        h   = self.tower(h)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, out_dim: int, n_classes: int, hidden: int,
                 embed_dim: int, latent_dim: int, n_blocks: int, dropout: float):
        super().__init__()
        self.label_embed = nn.Embedding(n_classes, embed_dim)
        self.stem = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, hidden, bias=False),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.tower = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head  = nn.Linear(hidden, out_dim)

    def forward(self, z, y):
        e   = self.label_embed(y)
        h   = self.stem(torch.cat([z, e], dim=1))
        h   = self.tower(h)
        # tanh maps to (-1,1) then scale to (-CLIP_VAL, CLIP_VAL)
        return torch.tanh(self.head(h)) * CLIP_VAL


class CVAE(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = HIDDEN_DIM,
                 embed_dim: int = EMBED_DIM, latent_dim: int = LATENT_DIM,
                 n_blocks: int = N_BLOCKS, dropout: float = DROPOUT):
        super().__init__()
        self.encoder = Encoder(in_dim, n_classes, hidden, embed_dim, latent_dim, n_blocks, dropout)
        self.decoder = Decoder(in_dim, n_classes, hidden, embed_dim, latent_dim, n_blocks, dropout)
        self.latent_dim = latent_dim
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # deterministic at eval

    def forward(self, x, y):
        mu, logvar = self.encoder(x, y)
        z          = self.reparameterize(mu, logvar)
        x_hat      = self.decoder(z, y)
        return x_hat, mu, logvar

    @torch.no_grad()
    def generate(self, y: torch.Tensor, n: int | None = None) -> torch.Tensor:
        """Sample n points conditioned on class labels y."""
        self.eval()
        if n is not None:
            y = y.repeat(n)
        z = torch.randn(len(y), self.latent_dim, device=next(self.parameters()).device)
        return self.decoder(z, y)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════════
#  LOSS
# ═══════════════════════════════════════════════════════════════════════════════

def elbo_loss(x, x_hat, mu, logvar, beta: float):
    recon = F.mse_loss(x_hat, x, reduction="mean")
    # KL divergence: -0.5 * mean(1 + log_var - mu^2 - var)
    kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl


def kl_beta(epoch: int) -> float:
    """Linear anneal from 0 to 1 over ANNEAL_EPOCHS."""
    return min(1.0, epoch / max(ANNEAL_EPOCHS, 1))


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def run_epoch(model, loader, optimizer, beta: float):
    training = optimizer is not None
    model.train(training)

    tot_loss = tot_recon = tot_kl = n = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            if training:
                optimizer.zero_grad(set_to_none=True)

            x_hat, mu, logvar = model(X, y)
            loss, recon, kl   = elbo_loss(X, x_hat, mu, logvar, beta)

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            bs = len(y)
            tot_loss  += loss.item()  * bs
            tot_recon += recon.item() * bs
            tot_kl    += kl.item()    * bs
            n         += bs

    return tot_loss / n, tot_recon / n, tot_kl / n


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_curves(history: dict, out_dir: Path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    keys = [
        ("train_loss",  "val_loss",  "ELBO Loss"),
        ("train_recon", "val_recon", "Reconstruction (MSE)"),
        ("train_kl",    "val_kl",    "KL Divergence"),
    ]
    for ax, (tr_k, vl_k, title) in zip(axes, keys):
        ax.plot(epochs, history[tr_k], label="Train", linewidth=1.8)
        ax.plot(epochs, history[vl_k], label="Val",   linewidth=1.8)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle("CVAE — Training Progress", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "vae_training_curves.png", dpi=150)
    plt.close()
    log.info("Saved outputs/vae_training_curves.png")


@torch.no_grad()
def collect_val_stats(model, loader):
    """Return (mu vectors, per-sample recon MSE, integer labels) for a val loader."""
    model.eval()
    mus, errors, ys = [], [], []
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        mu, _         = model.encoder(X, y)
        x_hat, _, _   = model(X, y)
        err = F.mse_loss(x_hat, X, reduction="none").mean(dim=1)
        mus.append(mu.cpu().numpy())
        errors.append(err.cpu().numpy())
        ys.append(y.cpu().numpy())
    return np.concatenate(mus), np.concatenate(errors), np.concatenate(ys)


def plot_latent_tsne(mus, labels, classes, out_dir):
    N   = min(6000, len(mus))
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(mus), N, replace=False)
    log.info(f"Running t-SNE on {N} latent vectors …")
    emb = TSNE(n_components=2, random_state=SEED, perplexity=40,
               n_iter=500, init="pca").fit_transform(mus[idx])

    fig, ax = plt.subplots(figsize=(13, 9))
    cmap = plt.get_cmap("tab20", len(classes))
    for i, cls in enumerate(classes):
        mask = labels[idx] == i
        if mask.sum() == 0:
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1], s=7, alpha=0.5,
                   color=cmap(i), label=cls, linewidths=0)
    ax.set_title("CVAE — t-SNE of Latent Space (val set, mu vectors)",
                 fontsize=13, fontweight="bold")
    ax.legend(markerscale=3, fontsize=7, ncol=2, loc="best",
              framealpha=0.7, edgecolor="none")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_dir / "vae_latent_tsne.png", dpi=150)
    plt.close()
    log.info("Saved outputs/vae_latent_tsne.png")


def plot_per_class_recon(errors, labels, classes, out_dir):
    order = sorted(range(len(classes)),
                   key=lambda i: np.median(errors[labels == i]))
    data  = [errors[labels == i] for i in order]
    names = [classes[i] for i in order]

    fig, ax = plt.subplots(figsize=(14, max(8, len(classes) * 0.45)))
    ax.boxplot(
        data, vert=False, labels=names,
        flierprops=dict(marker=".", markersize=2, alpha=0.3),
        patch_artist=True,
        boxprops=dict(facecolor="mediumseagreen", alpha=0.6),
    )
    ax.set_xlabel("Reconstruction MSE", fontsize=11)
    ax.set_title("CVAE — Per-Class Reconstruction Error (val set)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "vae_per_class_recon.png", dpi=150)
    plt.close()
    log.info("Saved outputs/vae_per_class_recon.png")


def plot_generated_vs_real(model, X_val, y_val, classes, out_dir):
    """Overlay real vs CVAE-generated histograms for the 6 highest-variance features."""
    top6 = np.argsort(X_val.var(axis=0))[-6:][::-1]
    feat_names = [FEATURE_COLS[i] for i in top6]

    y_t = torch.from_numpy(y_val).to(DEVICE)
    with torch.no_grad():
        model.eval()
        z     = torch.randn(len(y_val), model.latent_dim, device=DEVICE)
        X_gen = model.decoder(z, y_t).cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, fi, fname in zip(axes.flatten(), top6, feat_names):
        ax.hist(X_val[:, fi],  bins=60, density=True, alpha=0.6,
                color="steelblue", label="Real")
        ax.hist(X_gen[:, fi],  bins=60, density=True, alpha=0.6,
                color="tomato",    label="Generated")
        ax.set_title(fname, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("CVAE — Generated vs Real Feature Distributions (top-6 by variance)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "vae_generated_vs_real.png", dpi=150)
    plt.close()
    log.info("Saved outputs/vae_generated_vs_real.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 70)
    log.info("CICIoT2023 — Conditional VAE Training")
    log.info(f"Device  : {DEVICE}")
    log.info(f"PyTorch : {torch.__version__}")
    if DEVICE.type == "cuda":
        log.info(f"GPU     : {torch.cuda.get_device_name(0)}")
    log.info("=" * 70)

    # ── 1. Load attack-only data ──────────────────────────────────────────────
    X, y, classes = load_attack_data(DATA_DIR, MAX_SAMPLES_PER_CLASS)

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    # Replace inf/nan in-place to avoid a copy.
    X = np.where(np.isfinite(X), X, np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))

    scaler     = RobustScaler()
    le         = LabelEncoder()
    le.classes_ = classes

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=SEED
    )
    del X, y   # full arrays no longer needed

    X_tr  = np.clip(scaler.fit_transform(X_tr),  -CLIP_VAL, CLIP_VAL).astype(np.float32)
    X_val = np.clip(scaler.transform(X_val),      -CLIP_VAL, CLIP_VAL).astype(np.float32)

    n_features = X_tr.shape[1]
    n_classes  = len(classes)
    log.info(f"Features: {n_features}  |  Attack classes: {n_classes}")
    log.info(f"Train: {len(X_tr):,}  Val: {len(X_val):,}")

    nw = min(4, os.cpu_count() or 1)
    kw = dict(num_workers=nw, pin_memory=(DEVICE.type == "cuda"), persistent_workers=(nw > 0))
    train_loader = DataLoader(AttackDataset(X_tr,  y_tr),  batch_size=BATCH_SIZE, shuffle=True,  drop_last=True, **kw)
    val_loader   = DataLoader(AttackDataset(X_val, y_val), batch_size=BATCH_SIZE * 2, shuffle=False, **kw)

    # ── 3. Model ──────────────────────────────────────────────────────────────
    model = CVAE(in_dim=n_features, n_classes=n_classes).to(DEVICE)
    log.info(f"CVAE params: {model.count_params():,}")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)

    # ── 4. Training loop ──────────────────────────────────────────────────────
    history = {k: [] for k in ("train_loss", "val_loss", "train_recon", "val_recon", "train_kl", "val_kl")}
    best_val_loss = float("inf")
    best_state    = None
    patience_cnt  = 0

    for epoch in range(1, EPOCHS + 1):
        beta = kl_beta(epoch)
        t0   = time.time()

        tr_loss, tr_recon, tr_kl = run_epoch(model, train_loader, optimizer, beta)
        vl_loss, vl_recon, vl_kl = run_epoch(model, val_loader,   None,      beta)
        scheduler.step()

        for k, v in zip(
            ("train_loss", "val_loss", "train_recon", "val_recon", "train_kl", "val_kl"),
            (tr_loss, vl_loss, tr_recon, vl_recon, tr_kl, vl_kl),
        ):
            history[k].append(v)

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
            f"Epoch {epoch:3d}/{EPOCHS}  β={beta:.3f}  "
            f"tr_loss={tr_loss:.5f} (recon={tr_recon:.5f} kl={tr_kl:.4f})  "
            f"vl_loss={vl_loss:.5f}  "
            f"{time.time() - t0:.1f}s{flag}"
        )

        if patience_cnt >= EARLY_STOP_PATIENCE:
            log.info(f"Early stopping at epoch {epoch}.")
            break

    # ── 5. Save checkpoint ────────────────────────────────────────────────────
    if best_state:
        model.load_state_dict(best_state)

    ckpt_path = OUTPUT_DIR / "best_vae.pt"
    torch.save(
        {
            "model_state": best_state,
            "classes":     classes,
            "scaler":      scaler,
            "label_encoder": le,
            "config": {
                "in_dim":     n_features,
                "n_classes":  n_classes,
                "hidden":     HIDDEN_DIM,
                "embed_dim":  EMBED_DIM,
                "latent_dim": LATENT_DIM,
                "n_blocks":   N_BLOCKS,
                "dropout":    DROPOUT,
            },
        },
        ckpt_path,
    )
    log.info(f"CVAE checkpoint saved → {ckpt_path}")

    plot_curves(history, OUTPUT_DIR)

    # ── 7. Post-training visualisations ──────────────────────────────────────
    log.info("Generating post-training plots …")
    mus, val_errors, val_labels = collect_val_stats(model, val_loader)
    plot_latent_tsne(mus, val_labels, classes, OUTPUT_DIR)
    plot_per_class_recon(val_errors, val_labels, classes, OUTPUT_DIR)
    plot_generated_vs_real(model, X_val, y_val, classes, OUTPUT_DIR)

    log.info(f"All outputs written to: {OUTPUT_DIR.resolve()}")
    log.info("Done.")


if __name__ == "__main__":
    main()
