# ATML Assignment 0 — Representation Learning & Model Internals

EE-5102 / CS-6304 · Advanced Topics in Machine Learning

## Repository Overview

This repository contains the complete implementation and empirical analysis for ATML Assignment 0, covering four core representational paradigms in modern deep learning:
1. **Task 1 — ResNet-152:** Transfer learning, residual skip connection ablation, feature hierarchy evolution, and representation alignment.
2. **Task 2 — Vision Transformer (ViT):** Attention visualization, CLS token attention maps, patch-masking degradation robustness, and linear probing (CLS vs. mean-pooling).
3. **Task 3 — CLIP:** Zero-shot classification on STL-10 under prompt strategies, image-text modality gap analysis, and Procrustes alignment.
4. **Task 4 — Variational Autoencoder (VAE):** Generative modeling on MNIST, ELBO optimization, latent space geometry, and comparison with Doersch (2016).

---

## Directory Structure

```
atml-assignment0/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── report/                       # NeurIPS LaTeX report and references
│   ├── report.tex
│   ├── neurips_2024.sty
│   └── references.bib
├── task1_resnet152/              # Task 1 source, notebooks, results
├── task2_vit/                    # Task 2 source, notebooks, results
├── task3_clip/                   # Task 3 source, notebooks, results
├── task4_vae/                    # Task 4 source, notebooks, results
└── data/                         # Datasets (gitignored)
```

---

## Quick Setup & Execution

### 1. Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Running Tasks
Each task can be explored directly through its modular scripts or step-by-step Jupyter notebooks:
- **Task 1:** Navigate to `task1_resnet152/notebooks/` and run notebooks 1–5.
- **Task 2:** Navigate to `task2_vit/notebooks/` and run notebooks 1–4.
- **Task 3:** Navigate to `task3_clip/notebooks/` and run notebooks 1–3.
- **Task 4:** Navigate to `task4_vae/notebooks/` and run notebooks 1–3.
