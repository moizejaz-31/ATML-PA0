# Task 4 — Variational Autoencoder (VAE)

## Objectives & Key Findings

1. **ELBO Optimization:**
   - Trained MLP VAE on MNIST with Bernoulli reconstruction loss (BCE) and analytical Gaussian KL divergence.
   - ELBO steadily decreases over training epochs as reconstruction fidelity and latent regularization balance out.

2. **Latent Space & Sampling:**
   - 2D latent space plots display smooth, continuous transitions across digit classes.
   - Sampling directly from prior $z \sim \mathcal{N}(0, I)$ generates realistic, diverse handwritten digits.

3. **Comparison with Doersch (2016):**
   - Architecture choices (MLP vs. Conv), Bernoulli output assumptions, and latent dimensionality ($d=2$ vs higher dimensions) match theoretical formulations in Doersch's VAE tutorial.
