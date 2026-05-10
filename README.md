# IoT Cyber Security — CICIoT2023 Intrusion Detection

Three complementary models for detecting and generating IoT network intrusions using the [CICIoT2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html) dataset (34 traffic classes: 33 attack types + benign).

**Dataset layout expected:**

```
MERGED_CSV/          ← directory of *.csv files from CICIoT2023
outputs/             ← all model checkpoints and plots are written here
```

---

## Model 1 — Deep Residual MLP Classifier (`train.py`)

### What it does
Supervised multi-class classification of network flows into one of 34 known traffic classes (benign + 33 attack types).

### Architecture
```
Input (39 features)
  └→ Stem: Linear(39→512) + BatchNorm + GELU + Dropout(0.15)
       └→ ResBlock × 6  [512-dim, pre-activation: BN→GELU→Linear→BN→GELU→Drop→Linear + skip]
            └→ Head: BN → Linear(512→256) → GELU → Dropout → Linear(256→34)
```

- **Loss:** Multi-class Focal Loss (γ=2) + inverse-frequency class weights + label smoothing (0.05) — handles severe class imbalance across 34 classes
- **Optimizer:** AdamW + OneCycleLR (8% warmup, cosine anneal)
- **Mixed precision:** AMP on GPU

### How to run
```bash
# Dataset CSVs must be in ./MERGED_CSV/
python train.py
```

### Outputs (`outputs/`)
| File | Description |
|------|-------------|
| `best_model.pt` | Model checkpoint (weights + class list + config) |
| `train.log` | Full training log |
| `training_curves.png` | Train/val loss and accuracy per epoch |
| `confusion_matrix.png` | Normalised confusion matrix on test set |
| `per_class_f1.png` | F1-score bar chart for all 34 classes |
| `class_distribution.png` | Training set class distribution (log scale) |
| `classification_report.txt` | Per-class precision/recall/F1 + macro averages |

---

## Model 2 — Autoencoder Anomaly Detector (`ae_train.py` + `ae_detect.py`)

### What it does
Unsupervised anomaly detection that flags **novel/unknown attacks** not seen during training, then hands known traffic to the MLP classifier (two-stage pipeline in `ae_detect.py`).

### Architecture
```
Encoder:  39 → 256 → 128 → 64 → 16  (bottleneck)
Decoder:  16 →  64 → 128 → 256 → 39

Each layer: AEBlock = Linear + BatchNorm + GELU + Dropout(0.1) + skip connection
Final decoder layer: plain Linear (no activation — output lives in unbounded scaled space)
```

- **Loss:** MSE reconstruction loss
- **Threshold:** calibrated at the 99th percentile of validation-set reconstruction errors; samples above threshold are flagged as `UNKNOWN / NOVEL ATTACK`
- **Two-stage pipeline (`ae_detect.py`):** AE gates first; samples below threshold go to the MLP for classification

### How to run
```bash
# Train the autoencoder
python ae_train.py

# Run the two-stage pipeline (requires ae_model.pt + best_model.pt)
python ae_detect.py
```

`ae_detect.py` also runs a **Leave-One-Class-Out (LOCO) evaluation** — each attack class is withheld in turn and the AE's ability to flag it as anomalous (AUROC + detection rate) is measured.

### Outputs (`outputs/`)
| File | Description |
|------|-------------|
| `ae_model.pt` | AE checkpoint (weights + threshold + scaler stats + config) |
| `ae_train.log` | Training log |
| `ae_training_curves.png` | Train/val MSE per epoch |
| `ae_error_dist_val.png` | Reconstruction error distribution on val set + threshold line |
| `ae_error_dist_test.png` | Same for test set |
| `ae_per_class_errors.png` | Box-plot of reconstruction error per class (val set) |
| `ae_loco_evaluation.png` | AUROC + detection rate for each withheld attack class |
| `ae_score_histogram.png` | Known vs. unknown reconstruction error distributions |

---

## Model 3 — Conditional VAE Attack Generator (`vae_train.py` + `vae_generate.py`)

### What it does
Generates realistic synthetic attack traffic conditioned on attack class — useful for stress testing, data augmentation, and red-teaming scenarios. Trained only on attack classes (benign excluded).

### Architecture
```
Encoder: [x (39) || class_embed (64)] → Linear(103→256) + LayerNorm + GELU
           → ResBlock × 4 [256-dim, LayerNorm variant]
           → mu (64), log_var (64)

Decoder: [z (64) || class_embed (64)] → Linear(128→256) + LayerNorm + GELU
           → ResBlock × 4 [256-dim]
           → Linear(256→39) → tanh × 20  (clips to ±20, matching the scaler range)
```

- **Loss:** ELBO = MSE reconstruction + β·KL divergence; β anneals 0→1 over the first 25 epochs to prevent posterior collapse
- **Optimizer:** AdamW + CosineAnnealingLR
- **Generation filter:** `vae_generate.py` rejects samples where the MLP classifier disagrees with the intended attack class, improving quality

### How to run
```bash
# Train the CVAE (attack classes only)
python vae_train.py

# Generate synthetic attack flows
python vae_generate.py                                   # all attack classes, 2000 samples each
python vae_generate.py --classes "DDoS_ICMP_Flood" "XSS"
python vae_generate.py --n-per-class 5000 --no-tsne
python vae_generate.py --no-filter                       # skip MLP quality filter
```

### Outputs
| Location | File | Description |
|----------|------|-------------|
| `outputs/` | `best_vae.pt` | CVAE checkpoint (weights + scaler + label encoder + config) |
| `outputs/` | `vae_train.log` | Training log |
| `outputs/` | `vae_training_curves.png` | ELBO, reconstruction, and KL divergence per epoch |
| `outputs/` | `vae_generate.log` | Generation run log |
| `outputs/` | `vae_acceptance_rates.png` | MLP acceptance rate per attack class |
| `outputs/` | `vae_feature_distributions.png` | Real vs. synthetic feature histograms |
| `outputs/` | `vae_tsne.png` | t-SNE of real vs. synthetic samples |
| `outputs/generated/` | `<ClassName>.csv` | One CSV per attack class, inverse-transformed to original feature space, with a `Label` column |

---

## Recommended Run Order

```
1. python train.py       # train MLP classifier  →  outputs/best_model.pt
2. python ae_train.py    # train AE detector      →  outputs/ae_model.pt
3. python ae_detect.py   # evaluate two-stage pipeline
4. python vae_train.py   # train CVAE generator   →  outputs/best_vae.pt
5. python vae_generate.py # generate synthetic attacks → outputs/generated/*.csv
```

Steps 1–2 can run independently; step 3 requires both. Step 5 requires step 4, and optionally uses `best_model.pt` from step 1 as a quality filter.

---

## Features (39 total)

Network flow statistics used by all three models:

`Header_Length`, `Protocol Type`, `Time_To_Live`, `Rate`, TCP flag counts (`fin/syn/rst/psh/ack/ece/cwr`), aggregate flag counts (`ack_count/syn_count/fin_count/rst_count`), protocol indicators (`HTTP/HTTPS/DNS/Telnet/SMTP/SSH/IRC/TCP/UDP/DHCP/ARP/ICMP/IGMP/IPv/LLC`), flow statistics (`Tot sum/Min/Max/AVG/Std/Tot size/IAT/Number/Variance`).

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirents.txt
```

GPU with CUDA 12 is recommended. All models fall back to CPU automatically.
