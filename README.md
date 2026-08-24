# ATML Assignment 0 — Representation Learning & Model Internals

**Course:** EE-5102 / CS-6304 · Advanced Topics in Machine Learning  
**Institution:** Lahore University of Management Sciences (LUMS)

---

## Overview

This repository contains the complete implementation and empirical analysis for ATML Programming Assignment 0, covering four core representational paradigms in modern deep learning:

| Task | Model | Focus |
|------|-------|-------|
| **Task 1** | ResNet-152 | Transfer learning, residual ablation, feature hierarchies, representation alignment |
| **Task 2** | Vision Transformer (ViT-Base) | Attention maps, patch-masking robustness, CLS vs. mean-pooling linear probes |
| **Task 3** | CLIP (ViT-B/32) | Zero-shot classification on STL-10, modality gap analysis, Procrustes alignment |
| **Task 4** | Variational Autoencoder | ELBO training on MNIST, latent space geometry, generation, comparison with Doersch (2016) |

---

## Repository Structure

```
atml-assignment0/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package installer (pip install -e .)
├── LICENSE
│
├── task1_resnet152/
│   ├── src/
│   │   ├── model.py                   # ResNet-152 model loading & head replacement
│   │   ├── train.py                   # Training loop with caching & early stopping
│   │   ├── data.py                    # CIFAR-10 data loading & transforms
│   │   ├── hooks.py                   # Forward hooks for intermediate activations
│   │   ├── residual_ablation.py       # Skip connection ablation logic
│   │   └── visualize.py              # t-SNE, UMAP, CKA visualization utilities
│   ├── notebooks/
│   │   ├── 1_baseline_finetune.ipynb  # Pretrained vs. scratch training comparison
│   │   ├── 2_residual_ablation.ipynb  # Skip connection removal experiments
│   │   ├── 3_feature_hierarchies.ipynb # Layer-wise feature evolution & CKA
│   │   ├── 4_transfer_learning.ipynb  # Unfreezing strategies comparison
│   │   └── 5_optional_experiments.ipynb # Augmentation, schedulers, layer-wise LR
│   └── README.md
│
├── task2_vit/
│   ├── src/
│   │   ├── load_vit.py                # ViT-Base model loading & preprocessing
│   │   ├── attention.py               # Attention extraction, rollout & head entropy
│   │   ├── patch_masking.py           # Random & structured center patch masking
│   │   └── linear_probe.py           # Feature extraction & logistic regression probes
│   ├── notebooks/
│   │   ├── 1_classification.ipynb     # Pretrained ViT-Base inference on sample images
│   │   ├── 2_attention_maps.ipynb     # CLS attention, rollout, head entropy analysis
│   │   ├── 3_patch_masking.ipynb      # Robustness to random vs. center masking
│   │   └── 4_cls_vs_meanpool.ipynb    # CLS token vs. mean-pooled linear probing
│   ├── images/                        # Sample test images (golden retriever, tabby cat, sports car)
│   └── README.md
│
├── task3_clip/
│   ├── src/
│   │   ├── zeroshot.py                # STL-10 dataset, prompt strategies & zero-shot eval
│   │   ├── embeddings.py              # CLIP image/text feature extraction
│   │   ├── modality_gap.py            # t-SNE visualization & gap metrics computation
│   │   └── procrustes.py             # Orthogonal Procrustes alignment & aligned evaluation
│   ├── notebooks/
│   │   ├── 1_zeroshot_stl10.ipynb     # 4 prompt strategies evaluated on 8k test images
│   │   ├── 2_modality_gap.ipynb       # Modality gap visualization & quantification
│   │   └── 3_bridging_gap.ipynb       # Procrustes alignment & accuracy improvement
│   └── README.md
│
├── task4_vae/
│   ├── src/
│   │   ├── model.py                   # VAE encoder/decoder architecture (MLP)
│   │   ├── loss.py                    # ELBO loss (BCE + KL divergence)
│   │   ├── train.py                   # Training loop with epoch-level logging
│   │   └── analyze.py                # Latent visualization, interpolation & manifold grids
│   ├── notebooks/
│   │   ├── 1_train_vae.ipynb          # Train VAE on MNIST, loss curves
│   │   ├── 2_analyze_vae.ipynb        # Latent space, reconstruction, generation, interpolation
│   │   └── 3_compare_doersch.ipynb    # Comparison with Doersch (2016) tutorial & d=2 vs. d=10
│   └── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU recommended (all notebooks were run with CUDA)

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/atml-assignment0.git
cd atml-assignment0
```

### 2. Create virtual environment & install dependencies
```bash
python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

### 3. Run notebooks
Each task is self-contained. Navigate to the relevant `notebooks/` directory and run sequentially:

```bash
# Task 1: ResNet-152
jupyter notebook task1_resnet152/notebooks/

# Task 2: Vision Transformer
jupyter notebook task2_vit/notebooks/

# Task 3: CLIP
jupyter notebook task3_clip/notebooks/

# Task 4: VAE
jupyter notebook task4_vae/notebooks/
```

Datasets (CIFAR-10, STL-10, MNIST) are downloaded automatically on first run.

---

## Task Summaries

### Task 1 — ResNet-152 Inner Workings

| Notebook | Experiment | Key Finding |
|----------|-----------|-------------|
| `1_baseline_finetune` | Pretrained vs. scratch training | Pretrained reaches 95.8% in 5 epochs; scratch reaches 54.7% |
| `2_residual_ablation` | Disabling skip connections | Removing residual shortcuts degrades accuracy by up to 20% |
| `3_feature_hierarchies` | Layer-wise t-SNE/UMAP + CKA | Early layers encode edges/textures; deep layers show class separability |
| `4_transfer_learning` | Unfreezing strategies | Head-only 95.8%; unfreezing layer4 achieves 96.3%; random init 10% |
| `5_optional_experiments` | Augmentation, schedulers, LR decay | CosineAnnealing + augmentation pushes accuracy to 96.8% |

### Task 2 — Vision Transformer (ViT)

| Notebook | Experiment | Key Finding |
|----------|-----------|-------------|
| `1_classification` | Pretrained ViT-Base inference | Correct top-1 predictions with >90% confidence |
| `2_attention_maps` | CLS attention, rollout, head entropy | Rollout produces sharper object boundary heatmaps than single-layer attention |
| `3_patch_masking` | Random vs. center masking robustness | ViT maintains 100% accuracy at 50% random masking; 0% at 50% center masking |
| `4_cls_vs_meanpool` | Linear probing comparison | Both achieve 95.67% accuracy; cosine similarity between them is 0.82 |

### Task 3 — CLIP

| Notebook | Experiment | Key Finding |
|----------|-----------|-------------|
| `1_zeroshot_stl10` | 4 prompt strategies on 8,000 test images | Plain labels: 96.41%; Template: 95.73%; Descriptive: 94.16%; Ensemble: 95.69% |
| `2_modality_gap` | Modality gap visualization & metrics | Image and text embeddings occupy completely disjoint regions (centroid gap: 1.06) |
| `3_bridging_gap` | Orthogonal Procrustes alignment | Centroid gap drops from 1.06 to 0.13; accuracy improves by +0.7% to +2.0% |

### Task 4 — Variational Autoencoder

| Notebook | Experiment | Key Finding |
|----------|-----------|-------------|
| `1_train_vae` | ELBO training on MNIST (20 epochs) | Loss decreases from ~180 to ~142; KL stabilizes around 4.5 |
| `2_analyze_vae` | Latent space, reconstruction, generation | Smooth 2D manifold with continuous digit transitions; no dead zones |
| `3_compare_doersch` | Doersch (2016) comparison & d=2 vs. d=10 | Architecture matches tutorial; d=10 gives modestly sharper reconstructions |

---

## Dependencies

Core dependencies (see `requirements.txt` for versions):

| Package | Purpose |
|---------|---------|
| `torch`, `torchvision` | Deep learning framework & datasets |
| `transformers` | HuggingFace ViT model loading |
| `timm` | PyTorch Image Models |
| `open_clip_torch` | OpenCLIP model loading |
| `scikit-learn` | Linear probing, t-SNE, confusion matrices |
| `umap-learn` | UMAP dimensionality reduction |
| `matplotlib`, `seaborn` | Plotting & visualization |
| `scipy` | Orthogonal Procrustes alignment |
| `pillow` | Image loading & processing |

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
