#!/usr/bin/env python3
"""
CICIoT2023 — Two-Stage Anomaly + Classification Pipeline

Stage 1 (Autoencoder): compute reconstruction error.
         error > threshold  →  "UNKNOWN / NOVEL ATTACK"
Stage 2 (Residual MLP): classify into one of the 34 known classes.

Evaluation strategy for unknown detection:
  We simulate unknown attacks using a leave-one-class-out scheme:
  each known attack class is withheld in turn, the two-stage pipeline
  is run, and we measure how well the autoencoder flags those samples
  as anomalous (AUROC, detection rate at the calibrated threshold).
"""

import sys
import warnings
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import RobustScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
AE_CKPT    = OUTPUT_DIR / "ae_model.pt"
MLP_CKPT   = OUTPUT_DIR / "best_model.pt"

# ─── Hardware ────────────────────────────────────────────────────────────────
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
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS  (must match training scripts exactly)
# ═══════════════════════════════════════════════════════════════════════════════

class AEBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.fc   = nn.Linear(in_dim, out_dim, bias=False)
        self.bn   = nn.BatchNorm1d(out_dim)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.drop(self.act(self.bn(self.fc(x)))) + self.skip(x)


class Autoencoder(nn.Module):
    def __init__(self, in_dim, hidden_dims, latent_dim, dropout):
        super().__init__()
        enc, prev = [], in_dim
        for h in hidden_dims:
            enc.append(AEBlock(prev, h, dropout))
            prev = h
        enc += [nn.Linear(prev, latent_dim, bias=False), nn.BatchNorm1d(latent_dim)]
        self.encoder = nn.Sequential(*enc)

        dec, prev = [], latent_dim
        for h in reversed(hidden_dims):
            dec.append(AEBlock(prev, h, dropout))
            prev = h
        dec.append(nn.Linear(prev, in_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x):
        return F.mse_loss(self.forward(x), x, reduction="none").mean(dim=1)


class ResBlock(nn.Module):
    def __init__(self, dim, dropout):
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
            nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.tower = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(n_blocks)])
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

def load_autoencoder(path: Path) -> tuple[Autoencoder, float, RobustScaler, np.ndarray]:
    ckpt   = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg    = ckpt["config"]
    model  = Autoencoder(
        in_dim=cfg["in_dim"],
        hidden_dims=cfg["hidden_dims"],
        latent_dim=cfg["latent_dim"],
        dropout=cfg["dropout"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Reconstruct the scaler from saved statistics
    scaler         = RobustScaler()
    scaler.center_ = ckpt["scaler_mean"]
    scaler.scale_  = ckpt["scaler_scale"]

    threshold = float(ckpt["threshold"])
    classes   = ckpt["classes"]
    log.info(f"Autoencoder loaded from {path}  |  threshold={threshold:.6f}")
    return model, threshold, scaler, classes


def load_mlp(path: Path) -> tuple[ResidualMLP, np.ndarray]:
    ckpt  = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg   = ckpt["config"]
    model = ResidualMLP(
        in_dim=cfg["in_dim"],
        n_classes=cfg["n_classes"],
        hidden=cfg["hidden"],
        n_blocks=cfg["n_blocks"],
        dropout=cfg["dropout"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log.info(f"MLP loaded from {path}  |  classes={cfg['n_classes']}")
    return model, ckpt["classes"]


# ═══════════════════════════════════════════════════════════════════════════════
#  TWO-STAGE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

UNKNOWN_LABEL = "UNKNOWN / NOVEL ATTACK"

class TwoStagePipeline:
    """
    Combines the autoencoder anomaly detector with the MLP classifier.

    predict(X_raw) → list of str
      For each sample:
        1. Scale X with the AE's scaler and clip to [-20, 20].
        2. Compute reconstruction error.
        3. If error > threshold  →  UNKNOWN_LABEL
           Else                  →  run MLP, return known class name.
    """

    def __init__(self, ae: Autoencoder, threshold: float, scaler: RobustScaler,
                 mlp: ResidualMLP, ae_classes: np.ndarray, mlp_classes: np.ndarray):
        self.ae         = ae
        self.threshold  = threshold
        self.scaler     = scaler
        self.mlp        = mlp
        self.ae_classes = ae_classes
        self.mlp_classes = mlp_classes

    @torch.no_grad()
    def predict(self, X_raw: np.ndarray) -> list[str]:
        X_scaled = self.scaler.transform(X_raw).astype(np.float32)
        X_scaled = np.clip(X_scaled, -20.0, 20.0)
        X_t      = torch.from_numpy(X_scaled).to(DEVICE)

        errors   = self.ae.reconstruction_error(X_t).cpu().numpy()
        logits   = self.mlp(X_t).cpu().numpy()
        mlp_pred = logits.argmax(axis=1)

        results = []
        for i, err in enumerate(errors):
            if err > self.threshold:
                results.append(UNKNOWN_LABEL)
            else:
                results.append(str(self.mlp_classes[mlp_pred[i]]))
        return results

    @torch.no_grad()
    def anomaly_scores(self, X_raw: np.ndarray) -> np.ndarray:
        """Return raw reconstruction errors (useful for ROC analysis)."""
        X_scaled = self.scaler.transform(X_raw).astype(np.float32)
        X_scaled = np.clip(X_scaled, -20.0, 20.0)
        X_t      = torch.from_numpy(X_scaled).to(DEVICE)
        return self.ae.reconstruction_error(X_t).cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════════
#  LEAVE-ONE-CLASS-OUT EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def loco_evaluation(pipeline: TwoStagePipeline, X_all: np.ndarray,
                     y_all: np.ndarray, classes: np.ndarray, out_dir: Path):
    """
    Leave-One-Class-Out (LOCO) unknown-detection evaluation.

    For each class C (excluding Benign):
      - Treat all samples of class C as "unknown"
      - Treat all other samples as "known"
      - Measure AUROC: can the AE's reconstruction error separate C from known?

    This gives a realistic signal for how well the system detects truly
    novel attack patterns that were never seen during training.
    """
    log.info("\nRunning Leave-One-Class-Out unknown-detection evaluation …")

    # Identify the benign class index
    benign_idx = np.where(classes == "BenignTraffic")[0]
    if len(benign_idx) == 0:
        benign_idx = np.where(np.char.lower(classes.astype(str)) == "benign")[0]

    attack_classes = [i for i in range(len(classes))
                      if i not in (benign_idx.tolist() if len(benign_idx) else [])]

    aurocs = {}
    detection_rates = {}

    for cls_idx in attack_classes:
        cls_name = classes[cls_idx]
        unknown_mask = (y_all == cls_idx)
        known_mask   = ~unknown_mask

        if unknown_mask.sum() < 10:
            continue

        X_known   = X_all[known_mask]
        X_unknown = X_all[unknown_mask]

        scores_known   = pipeline.anomaly_scores(X_known)
        scores_unknown = pipeline.anomaly_scores(X_unknown)

        all_scores = np.concatenate([scores_known, scores_unknown])
        all_labels = np.concatenate([
            np.zeros(len(scores_known)),
            np.ones(len(scores_unknown)),
        ])

        try:
            auc = roc_auc_score(all_labels, all_scores)
        except ValueError:
            auc = 0.5

        det_rate = (scores_unknown > pipeline.threshold).mean()
        aurocs[cls_name]         = auc
        detection_rates[cls_name] = det_rate

        log.info(f"  {cls_name:<45s}  AUROC={auc:.3f}  DetRate={det_rate:.3f}")

    if aurocs:
        mean_auc = np.mean(list(aurocs.values()))
        mean_det = np.mean(list(detection_rates.values()))
        log.info(f"\n  Mean AUROC        : {mean_auc:.4f}")
        log.info(f"  Mean DetRate      : {mean_det:.4f}")

        plot_loco_results(aurocs, detection_rates, out_dir)

    return aurocs, detection_rates


def plot_loco_results(aurocs: dict, detection_rates: dict, out_dir: Path):
    classes  = sorted(aurocs, key=aurocs.get)
    aucs     = [aurocs[c] for c in classes]
    det_rates = [detection_rates[c] for c in classes]

    fig, axes = plt.subplots(1, 2, figsize=(18, max(8, len(classes) * 0.35)))

    for ax, vals, title, color in zip(
        axes,
        [aucs, det_rates],
        ["AUROC (unknown vs known)", f"Detection Rate @ threshold"],
        ["steelblue", "darkorange"],
    ):
        bars = ax.barh(classes, vals, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(0.5,  color="gray",  linestyle=":", alpha=0.6, linewidth=1)
        ax.axvline(0.9,  color="green", linestyle="--", alpha=0.5, linewidth=1)
        for bar, val in zip(bars, vals):
            ax.text(
                min(val + 0.01, 0.99),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8,
            )
        ax.set_xlim(0, 1.1)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle("Leave-One-Class-Out Unknown Detection Evaluation",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "ae_loco_evaluation.png", dpi=150)
    plt.close()
    log.info("Saved outputs/ae_loco_evaluation.png")


def plot_score_histogram(scores_known: np.ndarray, scores_unknown: np.ndarray,
                          threshold: float, out_dir: Path, cls_name: str = ""):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(scores_known,   bins=100, density=True, alpha=0.6,
            color="steelblue", label="Known traffic")
    ax.hist(scores_unknown, bins=100, density=True, alpha=0.6,
            color="tomato",    label=f"Unknown ({cls_name})" if cls_name else "Unknown")
    ax.axvline(threshold, color="black", linewidth=2, linestyle="--",
               label=f"Threshold = {threshold:.4f}")
    ax.set_xlabel("Reconstruction Error (MSE)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Anomaly Score Distribution: Known vs Unknown", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "ae_score_histogram.png", dpi=150)
    plt.close()
    log.info("Saved outputs/ae_score_histogram.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 70)
    log.info("CICIoT2023 — Two-Stage Anomaly + Classification Inference")
    log.info("=" * 70)

    # ── Load models ───────────────────────────────────────────────────────────
    if not AE_CKPT.exists():
        log.error(f"Autoencoder checkpoint not found: {AE_CKPT}")
        log.error("Run ae_train.py first.")
        sys.exit(1)

    ae, threshold, scaler, ae_classes = load_autoencoder(AE_CKPT)

    if not MLP_CKPT.exists():
        log.error(f"MLP checkpoint not found: {MLP_CKPT}")
        log.error("Run train.py first.")
        sys.exit(1)

    mlp, mlp_classes = load_mlp(MLP_CKPT)

    pipeline = TwoStagePipeline(ae, threshold, scaler, mlp, ae_classes, mlp_classes)

    # ── Demo: predict on a small synthetic batch ──────────────────────────────
    log.info("\nDemo: predicting on 5 random synthetic samples …")
    rng      = np.random.default_rng(0)
    X_demo   = rng.standard_normal((5, len(FEATURE_COLS))).astype(np.float32)
    preds    = pipeline.predict(X_demo)
    scores   = pipeline.anomaly_scores(X_demo)
    log.info(f"{'Sample':<8} {'Recon Error':<16} {'Prediction'}")
    log.info("-" * 70)
    for i, (score, pred) in enumerate(zip(scores, preds)):
        log.info(f"{i:<8} {score:<16.6f} {pred}")

    # ── Leave-one-class-out evaluation (requires data directory) ─────────────
    data_dir = BASE_DIR / "MERGED_CSV"
    if not data_dir.exists():
        log.warning(f"Data directory not found ({data_dir}). Skipping LOCO evaluation.")
        return

    log.info("\nLoading data for LOCO evaluation …")
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    SEED = 42
    MAX_PER_CLASS = 10_000   # lighter load for evaluation only

    csv_files = sorted(data_dir.glob("*.csv"))
    buckets: dict[str, list] = {}
    for fpath in csv_files:
        try:
            chunk = pd.read_csv(fpath, low_memory=False)
        except Exception:
            continue
        needed = FEATURE_COLS + ["Label"]
        if any(c not in chunk.columns for c in needed):
            continue
        chunk = chunk[needed].replace([float("inf"), float("-inf")], float("nan")).dropna(subset=FEATURE_COLS)
        for label, grp in chunk.groupby("Label", sort=False):
            if label not in buckets:
                buckets[label] = []
            already = sum(len(d) for d in buckets[label])
            remaining = MAX_PER_CLASS - already
            if remaining <= 0:
                continue
            buckets[label].append(grp.iloc[:remaining])

    parts = [pd.concat(v, ignore_index=True) for v in buckets.values()]
    df    = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=SEED)
    le    = LabelEncoder()
    y_all = le.fit_transform(df["Label"].values)
    X_all = df[FEATURE_COLS].values.astype(np.float32)
    loco_classes = le.classes_

    loco_evaluation(pipeline, X_all, y_all, loco_classes, OUTPUT_DIR)

    # Score histogram using the class with the highest detection rate as "unknown"
    scores_all = pipeline.anomaly_scores(X_all)
    # Pick the class with the worst (lowest) detection rate for illustration
    benign_mask = np.isin(y_all, np.where(np.char.lower(loco_classes.astype(str)) == "benigntraffic")[0])
    plot_score_histogram(
        scores_all[benign_mask],
        scores_all[~benign_mask],
        threshold,
        OUTPUT_DIR,
        cls_name="all attacks",
    )

    log.info(f"\nAll outputs written to: {OUTPUT_DIR.resolve()}")
    log.info("Done.")


if __name__ == "__main__":
    main()
