import torch
import torch.nn as nn
import torch.nn.functional as F

def vae_loss_function(recon_x, x, mu, logvar):
    """
    Negative ELBO = Reconstruction Loss (BCE) + KL Divergence.
    """
    x_flat = x.view(-1, 784)
    # Reconstruction loss (Bernoulli likelihood)
    BCE = F.binary_cross_entropy(recon_x, x_flat, reduction='sum')

    # KL Divergence: 0.5 * sum(mu^2 + sigma^2 - log(sigma^2) - 1)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = BCE + KLD
    return total_loss, BCE, KLD
