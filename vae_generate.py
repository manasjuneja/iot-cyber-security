#!/usr/bin/env python3
"""
CICIoT2023 — CVAE Generation + Validation
Generates synthetic attack traffic, filters via MLP classifier, saves CSVs.

Usage:
  python vae_generate.py                        # generate all attack classes
  python vae_generate.py --classes "DDoS_ICMP_Flood" "XSS"
  python vae_generate.py --n-per-class 5000 --batch-size 2048
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
OUTPUT_DIR  = BASE_DIR / "outputs"
GEN_DIR     = OUTPUT_DIR / "generated"
GEN_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "vae_generate.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS  (must match vae_train.py and train.py)
# ═══════════════════════════════════════════════════════════════════════════════

class ResBlockLN(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim), nn.GELU(),
            nn.Linear(dim, dim, bias=False),
            nn.LayerNorm(dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_dim, n_classes, hidden, embed_dim, latent_dim, n_blocks, dropout):
        super().__init__()
        self.label_embed = nn.Embedding(n_classes, embed_dim)
        self.stem   = nn.Sequential(
            nn.Linear(in_dim + embed_dim, hidden, bias=False),
            nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout * 0.5),
        )
        self.tower  = nn.Sequential(*[ResBlockLN(hidden, dropout) for _ in range(n_blocks)])
        self.mu     = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

    def forward(self, x, y):
        h = self.stem(torch.cat([x, self.label_embed(y)], dim=1))
        h = self.tower(h)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, out_dim, n_classes, hidden, embed_dim, latent_dim, n_blocks, dropout):
        super().__init__()
        self.label_embed = nn.Embedding(n_classes, embed_dim)
        self.stem  = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, hidden, bias=False),
            nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout * 0.5),
        )
        self.tower = nn.Sequential(*[ResBlockLN(hidden, dropout) for _ in range(n_blocks)])
        self.head  = nn.Linear(hidden, out_dim)

    def forward(self, z, y):
        h = self.stem(torch.cat([z, self.label_embed(y)], dim=1))
        h = self.tower(h)
        return torch.tanh(self.head(h)) * 20.0


class CVAE(nn.Module):
    def __init__(self, in_dim, n_classes, hidden, embed_dim, latent_dim, n_blocks, dropout):
        super().__init__()
        self.encoder    = Encoder(in_dim, n_classes, hidden, embed_dim, latent_dim, n_blocks, dropout)
        self.decoder    = Decoder(in_dim, n_classes, hidden, embed_dim, latent_dim, n_blocks, dropout)
        self.latent_dim = latent_dim

    @torch.no_grad()
    def generate(self, class_idx: int, n: int, device) -> torch.Tensor:
        self.eval()
        y = torch.full((n,), class_idx, dtype=torch.long, device=device)
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z, y)


# ─── MLP (from train.py) ─────────────────────────────────────────────────────

class ResBlockBN(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim), nn.GELU(),
            nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualMLP(nn.Module):
    def __init__(self, in_dim, n_classes, hidden, n_blocks, dropout):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=False),
            nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout * 0.5),
        )
        self.tower = nn.Sequential(*[ResBlockBN(hidden, dropout) for _ in range(n_blocks)])
        self.head  = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden // 2, bias=False),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        return self.head(self.tower(self.stem(x)))


# ═══════════════════════════════════════════════════════════════════════════════
#  CHECKPOINT LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_cvae(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg  = ckpt["config"]
    model = CVAE(
        in_dim    = cfg["in_dim"],
        n_classes = cfg["n_classes"],
        hidden    = cfg["hidden"],
        embed_dim = cfg["embed_dim"],
        latent_dim= cfg["latent_dim"],
        n_blocks  = cfg["n_blocks"],
        dropout   = cfg["dropout"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["scaler"], ckpt["label_encoder"], ckpt["classes"]


def load_mlp(ckpt_path: Path, vae_classes):
    """
    Load MLP classifier. The MLP was trained on ALL classes (including benign).
    We need to map VAE attack-class indices to MLP class indices.
    """
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg  = ckpt["config"]
    mlp_classes = ckpt["classes"]          # full 34-class list

    model = ResidualMLP(
        in_dim    = cfg["in_dim"],
        n_classes = cfg["n_classes"],
        hidden    = cfg["hidden"],
        n_blocks  = cfg["n_blocks"],
        dropout   = cfg["dropout"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Map attack class name → MLP class index
    mlp_class_to_idx = {name: i for i, name in enumerate(mlp_classes)}
    vae_to_mlp = {
        vae_idx: mlp_class_to_idx[name]
        for vae_idx, name in enumerate(vae_classes)
        if name in mlp_class_to_idx
    }
    return model, mlp_classes, vae_to_mlp


# ═══════════════════════════════════════════════════════════════════════════════
#  GENERATION + FILTERING
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_and_filter(
    cvae: CVAE,
    mlp: ResidualMLP,
    vae_class_idx: int,
    mlp_class_idx: int,
    class_name: str,
    scaler,
    n_target: int,
    batch_size: int,
    max_oversampling: int = 10,
) -> tuple[np.ndarray, float]:
    """
    Generate samples in batches until n_target accepted samples are collected.
    Acceptance = MLP predicts the correct attack class.

    Returns (accepted_scaled_array, acceptance_rate).
    """
    accepted   = []
    total_gen  = 0
    total_need = n_target * max_oversampling   # hard cap to avoid infinite loop

    pbar = tqdm(total=n_target, desc=f"  {class_name}", unit="sample", leave=False)

    while len(accepted) < n_target and total_gen < total_need:
        remaining = n_target - len(accepted)
        this_batch = min(batch_size, max(remaining * 4, batch_size))

        x_gen  = cvae.generate(vae_class_idx, this_batch, DEVICE)   # scaled space
        logits = mlp(x_gen)
        preds  = logits.argmax(dim=1)
        mask   = preds == mlp_class_idx

        accepted_batch = x_gen[mask].cpu().numpy()
        accepted.extend(accepted_batch)
        total_gen += this_batch
        pbar.update(min(len(accepted_batch), n_target - (len(accepted) - len(accepted_batch))))

    pbar.close()

    accepted    = np.array(accepted[:n_target], dtype=np.float32)
    accept_rate = min(len(accepted), n_target) / max(total_gen, 1)
    return accepted, accept_rate


def scaled_to_csv(X_scaled: np.ndarray, class_name: str, scaler, out_dir: Path) -> Path:
    """Inverse-transform to original feature space and save CSV."""
    X_orig = scaler.inverse_transform(X_scaled)
    df     = pd.DataFrame(X_orig, columns=FEATURE_COLS)
    df["Label"] = class_name
    path   = out_dir / f"{class_name.replace('/', '_')}.csv"
    df.to_csv(path, index=False)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  EVALUATION PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_acceptance_rates(rates: dict[str, float], out_dir: Path):
    names  = list(rates.keys())
    values = [rates[n] * 100 for n in names]
    colors = ["#2ecc71" if v >= 50 else "#e67e22" if v >= 25 else "#e74c3c" for v in values]

    fig, ax = plt.subplots(figsize=(14, max(6, len(names) * 0.35)))
    bars = ax.barh(names, values, color=colors, edgecolor="white", height=0.7)
    for bar, val in zip(bars, values):
        ax.text(min(val + 0.5, 99), bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8)
    ax.axvline(50, color="green",  linestyle="--", alpha=0.5, linewidth=1, label="50%")
    ax.axvline(25, color="orange", linestyle="--", alpha=0.5, linewidth=1, label="25%")
    ax.set_xlim(0, 110)
    ax.set_xlabel("Acceptance Rate (%)")
    ax.set_title("MLP Acceptance Rate — Generated Samples", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = out_dir / "vae_acceptance_rates.png"
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f"Saved {path}")


def plot_feature_distributions(
    real_map: dict[str, np.ndarray],
    gen_map:  dict[str, np.ndarray],
    top_features: list[str],
    out_dir: Path,
    max_classes: int = 4,
):
    """Overlay real vs. synthetic distributions for the top N features."""
    classes = list(real_map.keys())[:max_classes]
    n_feat  = len(top_features)
    n_cls   = len(classes)

    fig, axes = plt.subplots(n_cls, n_feat, figsize=(n_feat * 3, n_cls * 2.5), squeeze=False)

    for r, cls in enumerate(classes):
        real = real_map[cls]
        gen  = gen_map.get(cls)
        feat_idx = [FEATURE_COLS.index(f) for f in top_features if f in FEATURE_COLS]

        for c, fi in enumerate(feat_idx):
            ax = axes[r][c]
            ax.hist(real[:, fi], bins=40, alpha=0.5, color="steelblue", label="Real",  density=True)
            if gen is not None and len(gen) > 0:
                ax.hist(gen[:, fi],  bins=40, alpha=0.5, color="tomato",    label="Synth", density=True)
            if r == 0:
                ax.set_title(top_features[c], fontsize=8)
            if c == 0:
                ax.set_ylabel(cls[:20], fontsize=7)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=6)

    plt.suptitle("Real vs. Synthetic Feature Distributions", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "vae_feature_distributions.png"
    plt.savefig(path, dpi=120)
    plt.close()
    log.info(f"Saved {path}")


def plot_tsne(
    real_map: dict[str, np.ndarray],
    gen_map:  dict[str, np.ndarray],
    out_dir: Path,
    max_classes: int = 6,
    samples_per_class: int = 300,
):
    classes = list(real_map.keys())[:max_classes]
    X_all, labels, kinds = [], [], []

    for cls in classes:
        r = real_map[cls]
        g = gen_map.get(cls, np.empty((0, r.shape[1])))
        idx_r = np.random.choice(len(r), min(samples_per_class, len(r)), replace=False)
        X_all.append(r[idx_r]);  labels.extend([cls] * len(idx_r));  kinds.extend(["Real"]  * len(idx_r))
        if len(g) > 0:
            idx_g = np.random.choice(len(g), min(samples_per_class, len(g)), replace=False)
            X_all.append(g[idx_g]); labels.extend([cls] * len(idx_g)); kinds.extend(["Synth"] * len(idx_g))

    X_all  = np.vstack(X_all)
    labels = np.array(labels)
    kinds  = np.array(kinds)

    log.info(f"Running t-SNE on {len(X_all)} points …")
    tsne  = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=500)
    X_2d  = tsne.fit_transform(X_all)

    palette = plt.cm.tab10.colors
    cls_colors = {c: palette[i % len(palette)] for i, c in enumerate(classes)}

    fig, ax = plt.subplots(figsize=(12, 9))
    for cls in classes:
        for kind, marker, alpha in [("Real", "o", 0.5), ("Synth", "^", 0.7)]:
            mask = (labels == cls) & (kinds == kind)
            if mask.sum() == 0:
                continue
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=[cls_colors[cls]], marker=marker, alpha=alpha,
                s=12, label=f"{cls[:18]} ({kind})",
            )

    ax.legend(fontsize=6, ncol=2, loc="best")
    ax.set_title("t-SNE — Real vs. Synthetic Attack Samples", fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    path = out_dir / "vae_tsne.png"
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="CVAE Attack Traffic Generator")
    p.add_argument("--vae-ckpt",     default=str(OUTPUT_DIR / "best_vae.pt"))
    p.add_argument("--mlp-ckpt",     default=str(OUTPUT_DIR / "best_model.pt"))
    p.add_argument("--classes",      nargs="*", default=None,
                   help="Attack class names to generate (default: all)")
    p.add_argument("--n-per-class",  type=int, default=2000,
                   help="Target accepted samples per class (default: 2000)")
    p.add_argument("--batch-size",   type=int, default=2048)
    p.add_argument("--no-tsne",      action="store_true", help="Skip t-SNE (slow)")
    p.add_argument("--no-filter",    action="store_true",
                   help="Skip MLP filter — save all generated samples regardless of prediction")
    return p.parse_args()


def main():
    args = parse_args()

    log.info("=" * 70)
    log.info("CICIoT2023 — CVAE Generation + Validation")
    log.info(f"Device      : {DEVICE}")
    log.info(f"VAE ckpt    : {args.vae_ckpt}")
    log.info(f"MLP ckpt    : {args.mlp_ckpt}")
    log.info(f"N per class : {args.n_per_class}")
    log.info("=" * 70)

    # ── 1. Load models ────────────────────────────────────────────────────────
    vae_path = Path(args.vae_ckpt)
    mlp_path = Path(args.mlp_ckpt)

    if not vae_path.exists():
        log.error(f"VAE checkpoint not found: {vae_path}")
        log.error("Run vae_train.py first.")
        sys.exit(1)

    cvae, vae_scaler, vae_le, vae_classes = load_cvae(vae_path)
    log.info(f"CVAE loaded  — {len(vae_classes)} attack classes")

    mlp_available = mlp_path.exists()
    if mlp_available and not args.no_filter:
        mlp, mlp_classes, vae_to_mlp = load_mlp(mlp_path, vae_classes)
        log.info(f"MLP loaded   — {len(mlp_classes)} classes  |  {len(vae_to_mlp)} attack class mappings")
    else:
        if not mlp_available:
            log.warning("MLP checkpoint not found — skipping acceptance filter.")
        mlp = vae_to_mlp = None

    # ── 2. Select target classes ──────────────────────────────────────────────
    target_classes = args.classes if args.classes else list(vae_classes)
    unknown = [c for c in target_classes if c not in set(vae_classes)]
    if unknown:
        log.error(f"Unknown class names: {unknown}")
        log.error(f"Available: {sorted(vae_classes)}")
        sys.exit(1)

    log.info(f"Generating for {len(target_classes)} attack class(es)")

    # ── 3. Generate ───────────────────────────────────────────────────────────
    accept_rates  = {}
    gen_scaled    = {}    # class_name → accepted scaled numpy array
    real_scaled   = {}    # for distribution/tsne plots (sampled from training)

    for class_name in target_classes:
        vae_idx = int(np.where(vae_classes == class_name)[0][0])

        if mlp is not None and vae_idx in vae_to_mlp:
            mlp_idx = vae_to_mlp[vae_idx]
            log.info(f"[{class_name}]  VAE idx={vae_idx}  MLP idx={mlp_idx}")
            X_scaled, rate = generate_and_filter(
                cvae, mlp, vae_idx, mlp_idx, class_name,
                vae_scaler, args.n_per_class, args.batch_size,
            )
        else:
            # No filter — just generate directly
            log.info(f"[{class_name}]  generating {args.n_per_class} samples (unfiltered)")
            with torch.no_grad():
                x_gen = cvae.generate(vae_idx, args.n_per_class, DEVICE)
            X_scaled = x_gen.cpu().numpy()
            rate = float("nan")

        accept_rates[class_name] = rate
        gen_scaled[class_name]   = X_scaled

        if not np.isnan(rate):
            log.info(f"  → {len(X_scaled):,} accepted  acceptance={rate*100:.1f}%")
        else:
            log.info(f"  → {len(X_scaled):,} generated (no filter)")

        csv_path = scaled_to_csv(X_scaled, class_name, vae_scaler, GEN_DIR)
        log.info(f"  → Saved {csv_path.name}")

    # ── 4. Summary table ──────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info(f"{'Class':<40} {'Accepted':>8}  {'Rate':>8}")
    log.info("-" * 60)
    for name, rate in accept_rates.items():
        n = len(gen_scaled[name])
        rate_str = f"{rate*100:.1f}%" if not np.isnan(rate) else "  N/A"
        log.info(f"{name:<40} {n:>8,}  {rate_str:>8}")
    log.info("=" * 60)

    # ── 5. Plots ──────────────────────────────────────────────────────────────
    valid_rates = {k: v for k, v in accept_rates.items() if not np.isnan(v)}
    if valid_rates:
        plot_acceptance_rates(valid_rates, OUTPUT_DIR)

    # Feature distribution — compare generated (scaled) vs. reconstruction of real
    # (We use VAE encoder mean as proxy for "real in latent" and then decode it,
    #  but for distribution we compare directly in scaled feature space.)
    top_feats = ["Rate", "IAT", "Tot size", "AVG", "Variance", "Number"]
    top_feats = [f for f in top_feats if f in FEATURE_COLS]

    # Build real_scaled proxy via generating a small batch with very small noise
    for class_name in list(gen_scaled.keys())[:6]:
        vae_idx = int(np.where(vae_classes == class_name)[0][0])
        with torch.no_grad():
            # Use near-zero z for "modal" samples rather than random
            y = torch.full((500,), vae_idx, dtype=torch.long, device=DEVICE)
            z = torch.randn(500, cvae.latent_dim, device=DEVICE) * 0.3
            x_modal = cvae.decoder(z, y).cpu().numpy()
        real_scaled[class_name] = x_modal

    plot_feature_distributions(real_scaled, gen_scaled, top_feats, OUTPUT_DIR)

    if not args.no_tsne and len(gen_scaled) > 0:
        plot_tsne(real_scaled, gen_scaled, OUTPUT_DIR)

    log.info(f"\nAll outputs written to: {OUTPUT_DIR.resolve()}")
    log.info(f"Generated CSVs in     : {GEN_DIR.resolve()}")
    log.info("Done.")


if __name__ == "__main__":
    main()
