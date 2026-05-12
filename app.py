#!/usr/bin/env python3
"""
IoT Cybersecurity Demo — Flask backend
Three model endpoints:
  POST /api/classify  — MLP 34-class classifier
  POST /api/detect    — AE anomaly detection + MLP fallback
  POST /api/generate  — CVAE synthetic traffic generation
"""

import warnings
warnings.filterwarnings("ignore")

import io
import base64
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request
from sklearn.preprocessing import RobustScaler

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
MLP_CKPT   = OUTPUT_DIR / "best_model.pt"
AE_CKPT    = OUTPUT_DIR / "ae_model.pt"
VAE_CKPT   = OUTPUT_DIR / "best_vae.pt"

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

# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

class AEBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
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
            enc.append(AEBlock(prev, h, dropout)); prev = h
        enc += [nn.Linear(prev, latent_dim, bias=False), nn.BatchNorm1d(latent_dim)]
        self.encoder = nn.Sequential(*enc)
        dec, prev = [], latent_dim
        for h in reversed(hidden_dims):
            dec.append(AEBlock(prev, h, dropout)); prev = h
        dec.append(nn.Linear(prev, in_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x):
        return F.mse_loss(self.forward(x), x, reduction="none").mean(dim=1)


class ResBlockBN(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim), nn.GELU(),
            nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim, bias=False),
        )
    def forward(self, x): return x + self.block(x)


class ResidualMLP(nn.Module):
    def __init__(self, in_dim, n_classes, hidden, n_blocks, dropout):
        super().__init__()
        self.stem  = nn.Sequential(
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
    def forward(self, x): return self.head(self.tower(self.stem(x)))


class ResBlockLN(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim), nn.GELU(),
            nn.Linear(dim, dim, bias=False),
            nn.LayerNorm(dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim, bias=False),
        )
    def forward(self, x): return x + self.block(x)


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
        return self.mu(self.tower(h)), self.logvar(self.tower(h))


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
        return torch.tanh(self.head(self.tower(h))) * 20.0


class CVAE(nn.Module):
    def __init__(self, in_dim, n_classes, hidden, embed_dim, latent_dim, n_blocks, dropout):
        super().__init__()
        self.encoder    = Encoder(in_dim, n_classes, hidden, embed_dim, latent_dim, n_blocks, dropout)
        self.decoder    = Decoder(in_dim, n_classes, hidden, embed_dim, latent_dim, n_blocks, dropout)
        self.latent_dim = latent_dim

    @torch.no_grad()
    def generate(self, class_idx, n, device):
        self.eval()
        y = torch.full((n,), class_idx, dtype=torch.long, device=device)
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z, y)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_mlp():
    ckpt  = torch.load(MLP_CKPT, map_location=DEVICE, weights_only=False)
    cfg   = ckpt["config"]
    model = ResidualMLP(
        in_dim=cfg["in_dim"], n_classes=cfg["n_classes"],
        hidden=cfg["hidden"], n_blocks=cfg["n_blocks"], dropout=cfg["dropout"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["classes"]


def load_ae():
    ckpt  = torch.load(AE_CKPT, map_location=DEVICE, weights_only=False)
    cfg   = ckpt["config"]
    model = Autoencoder(
        in_dim=cfg["in_dim"], hidden_dims=cfg["hidden_dims"],
        latent_dim=cfg["latent_dim"], dropout=cfg["dropout"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    scaler         = RobustScaler()
    scaler.center_ = ckpt["scaler_mean"]
    scaler.scale_  = ckpt["scaler_scale"]
    return model, float(ckpt["threshold"]), scaler


def load_vae():
    ckpt  = torch.load(VAE_CKPT, map_location=DEVICE, weights_only=False)
    cfg   = ckpt["config"]
    model = CVAE(
        in_dim=cfg["in_dim"], n_classes=cfg["n_classes"],
        hidden=cfg["hidden"], embed_dim=cfg["embed_dim"],
        latent_dim=cfg["latent_dim"], n_blocks=cfg["n_blocks"], dropout=cfg["dropout"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["scaler"], ckpt["classes"]


# ═══════════════════════════════════════════════════════════════════════════════
#  FLASK APP
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)

print("Loading models …")
mlp_model, mlp_classes  = load_mlp()
ae_model, ae_threshold, ae_scaler = load_ae()
vae_model, vae_scaler, vae_classes = load_vae()
print(f"  MLP: {len(mlp_classes)} classes")
print(f"  AE : threshold={ae_threshold:.6f}")
print(f"  VAE: {len(vae_classes)} attack classes")
print("Models ready.")


def features_from_request(data: dict) -> np.ndarray:
    """Parse feature dict from JSON body into a (1, 39) float32 array."""
    row = [float(data.get(col, 0.0)) for col in FEATURE_COLS]
    return np.array([row], dtype=np.float32)


# ─── Chart palette (warm dark / amber) ───────────────────────────────────────
C_BG      = "#0A0905"
C_SURFACE = "#17150E"
C_BORDER  = "#252218"
C_TEXT    = "#DDD4BC"
C_MUTED   = "#5A5440"
C_AMBER   = "#C4832A"
C_AMBER2  = "#8B6020"
C_GREEN   = "#4D9970"
C_RED     = "#B84A4A"

def _apply_dark_style(fig, *axes):
    fig.patch.set_facecolor(C_BG)
    for ax in axes:
        ax.set_facecolor(C_SURFACE)
        for spine in ax.spines.values():
            spine.set_color(C_BORDER)
        ax.tick_params(colors=C_MUTED, labelsize=8)
        ax.xaxis.label.set_color(C_MUTED)
        ax.yaxis.label.set_color(C_MUTED)
        ax.title.set_color(C_TEXT)
        ax.title.set_fontsize(10)

def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


@app.route("/")
def index():
    return render_template("index.html",
                           mlp_classes=sorted(mlp_classes.tolist()),
                           vae_classes=sorted(vae_classes.tolist()))


# ── 1. MLP Classifier ────────────────────────────────────────────────────────

@app.route("/api/classify", methods=["POST"])
def classify():
    data  = request.get_json()
    X_raw = features_from_request(data)

    # Scale using AE scaler (same RobustScaler used in training)
    X_sc  = ae_scaler.transform(X_raw).astype(np.float32)
    X_sc  = np.clip(X_sc, -20.0, 20.0)
    X_t   = torch.from_numpy(X_sc).to(DEVICE)

    with torch.no_grad():
        logits = mlp_model(X_t)
        probs  = F.softmax(logits, dim=1).cpu().numpy()[0]

    top_k = 8
    top_idx   = np.argsort(probs)[::-1][:top_k]
    top_probs = probs[top_idx]
    top_names = [str(mlp_classes[i]) for i in top_idx]

    pred_class = top_names[0]
    confidence = float(top_probs[0])

    # Bar chart — dark amber theme
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bar_colors = [C_AMBER] + [C_AMBER2] * (top_k - 1)
    bars = ax.barh(top_names[::-1], top_probs[::-1] * 100,
                   color=bar_colors[::-1], edgecolor="none", height=0.55)
    for bar, val in zip(bars, top_probs[::-1]):
        ax.text(min(val * 100 + 1.0, 102), bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", va="center", fontsize=8, color=C_MUTED,
                fontfamily="monospace")
    ax.set_xlabel("Confidence (%)", fontsize=9)
    ax.set_title("Top-8 Predicted Classes", fontweight="normal", pad=12)
    ax.set_xlim(0, 108)
    ax.axvline(50, color=C_BORDER, linewidth=0.8, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    _apply_dark_style(fig, ax)
    plt.tight_layout(pad=1.2)
    chart = fig_to_b64(fig)

    return jsonify({
        "prediction": pred_class,
        "confidence": round(confidence * 100, 2),
        "top_classes": [{"name": n, "prob": round(float(p) * 100, 2)}
                        for n, p in zip(top_names, top_probs)],
        "chart": chart,
    })


# ── 2. AE Anomaly Detector ────────────────────────────────────────────────────

@app.route("/api/detect", methods=["POST"])
def detect():
    data  = request.get_json()
    X_raw = features_from_request(data)

    X_sc  = ae_scaler.transform(X_raw).astype(np.float32)
    X_sc  = np.clip(X_sc, -20.0, 20.0)
    X_t   = torch.from_numpy(X_sc).to(DEVICE)

    with torch.no_grad():
        recon_err = ae_model.reconstruction_error(X_t).item()
        logits    = mlp_model(X_t)
        probs     = F.softmax(logits, dim=1).cpu().numpy()[0]

    is_anomaly  = recon_err > ae_threshold
    mlp_pred    = str(mlp_classes[np.argmax(probs)])
    final_label = "UNKNOWN / NOVEL ATTACK" if is_anomaly else mlp_pred
    confidence  = float(np.max(probs))

    # Anomaly detection chart — dark amber theme
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    # Left: reconstruction error vs threshold
    ax = axes[0]
    max_val = max(recon_err, ae_threshold) * 1.7
    err_color = C_RED if is_anomaly else C_GREEN
    ax.barh(["Threshold", "Recon Error"],
            [ae_threshold, recon_err],
            color=[C_MUTED, err_color], edgecolor="none", height=0.35)
    ax.axvline(ae_threshold, color=C_AMBER, linestyle="--", linewidth=1.2,
               label=f"thresh = {ae_threshold:.5f}")
    ax.set_xlim(0, max_val)
    ax.set_title("Reconstruction Error", fontweight="normal", pad=10)
    legend = ax.legend(fontsize=7.5, facecolor=C_SURFACE, edgecolor=C_BORDER,
                       labelcolor=C_MUTED)
    ax.spines[["top", "right"]].set_visible(False)

    # Right: top-5 MLP class probabilities
    ax2 = axes[1]
    top5_idx   = np.argsort(probs)[::-1][:5]
    top5_probs = probs[top5_idx]
    top5_names = [str(mlp_classes[i]) for i in top5_idx]
    c_primary = C_AMBER if is_anomaly else C_GREEN
    bar_colors2 = [c_primary] + [C_AMBER2 if is_anomaly else C_MUTED] * 4
    ax2.barh(top5_names[::-1], top5_probs[::-1] * 100,
             color=bar_colors2[::-1], edgecolor="none", height=0.45)
    for i, (bar, val) in enumerate(zip(ax2.patches, top5_probs[::-1])):
        ax2.text(min(val * 100 + 0.8, 107), bar.get_y() + bar.get_height() / 2,
                 f"{val*100:.1f}%", va="center", fontsize=7.5, color=C_MUTED,
                 fontfamily="monospace")
    ax2.set_xlim(0, 112)
    ax2.set_title("MLP Classifier Scores", fontweight="normal", pad=10)
    ax2.spines[["top", "right"]].set_visible(False)

    _apply_dark_style(fig, axes[0], axes[1])
    plt.tight_layout(pad=1.4)
    chart = fig_to_b64(fig)

    return jsonify({
        "recon_error":  round(recon_err, 6),
        "threshold":    round(ae_threshold, 6),
        "is_anomaly":   bool(is_anomaly),
        "final_label":  final_label,
        "mlp_label":    mlp_pred,
        "confidence":   round(confidence * 100, 2),
        "chart":        chart,
    })


# ── 3. CVAE Generator ─────────────────────────────────────────────────────────

@app.route("/api/generate", methods=["POST"])
def generate():
    data       = request.get_json()
    class_name = data.get("class_name", "DDOS-ICMP_FLOOD")
    n_samples  = int(data.get("n_samples", 200))
    n_samples  = max(50, min(n_samples, 500))

    vae_idx = int(np.where(vae_classes == class_name)[0][0])

    with torch.no_grad():
        x_gen    = vae_model.generate(vae_idx, n_samples, DEVICE)
        x_scaled = x_gen.cpu().numpy()

    # Inverse-transform to original feature space
    x_orig = vae_scaler.inverse_transform(x_scaled)

    show_features = ["Rate", "IAT", "Tot size", "AVG", "Number", "Variance",
                     "Header_Length", "Time_To_Live"]
    show_features = [f for f in show_features if f in FEATURE_COLS]
    feat_idx      = [FEATURE_COLS.index(f) for f in show_features]

    # Earth-tone palette for histograms — dark theme
    palette = [C_AMBER, "#A06830", "#7A5828", C_GREEN,
               "#3A7A58", "#C4702A", C_AMBER2, "#5A8070"]

    fig, axes = plt.subplots(2, 4, figsize=(14, 5.5))
    axes = axes.flatten()

    for i, (fi, fname) in enumerate(zip(feat_idx, show_features)):
        ax = axes[i]
        vals = x_orig[:, fi]
        ax.hist(vals, bins=28, color=palette[i % len(palette)],
                alpha=0.9, edgecolor="none", linewidth=0)
        ax.set_title(fname, fontsize=8.5, fontweight="normal", pad=6)
        ax.tick_params(labelsize=6.5)
        ax.spines[["top", "right"]].set_visible(False)
        _apply_dark_style(fig, ax)

    for ax in axes[len(show_features):]:
        ax.set_visible(False)

    fig.suptitle(f"{class_name}  ·  {n_samples} synthetic samples",
                 fontsize=9.5, color=C_MUTED, y=1.0, fontfamily="monospace")
    plt.tight_layout(pad=1.2, h_pad=1.6, w_pad=1.2)
    chart = fig_to_b64(fig)

    # Summary stats for key features
    stats = {}
    for fi, fname in zip(feat_idx, show_features):
        vals = x_orig[:, fi]
        stats[fname] = {
            "mean":   round(float(np.mean(vals)), 4),
            "std":    round(float(np.std(vals)), 4),
            "min":    round(float(np.min(vals)), 4),
            "max":    round(float(np.max(vals)), 4),
        }

    return jsonify({
        "class_name": class_name,
        "n_generated": n_samples,
        "stats":       stats,
        "chart":       chart,
    })


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5050)
