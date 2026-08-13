import os
import torch
import torch.optim as optim

from tqdm.auto import tqdm
from task4_vae.src.loss import vae_loss_function


def train_vae(model, train_loader, epochs=20, lr=1e-3, device="cpu",
              save_dir=None):
    """
    Train the VAE using Adam optimizer and ELBO loss.
    Returns training history as a list of dicts.
    Optionally saves best checkpoint to save_dir.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = []
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        bce_total = 0.0
        kld_total = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for data, _ in pbar:
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(data)
            loss, bce, kld = vae_loss_function(recon_batch, data, mu, logvar)

            loss.backward()
            train_loss += loss.item()
            bce_total += bce.item()
            kld_total += kld.item()
            optimizer.step()
            pbar.set_postfix({'loss': loss.item() / len(data)})

        num_samples = len(train_loader.dataset)
        avg_loss = train_loss / num_samples
        avg_bce = bce_total / num_samples
        avg_kld = kld_total / num_samples

        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.2f} (BCE: {avg_bce:.2f}, KLD: {avg_kld:.2f})")
        history.append({"epoch": epoch, "loss": avg_loss, "bce": avg_bce, "kld": avg_kld})

        # Save best model
        if save_dir and avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, "vae_mnist.pth"))

    return history


def evaluate_vae(model, test_loader, device="cpu"):
    """
    Evaluate VAE on test set, returning average loss, BCE, and KLD.
    """
    model.eval()
    model.to(device)
    total_loss = 0.0
    total_bce = 0.0
    total_kld = 0.0

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon, mu, logvar = model(data)
            loss, bce, kld = vae_loss_function(recon, data, mu, logvar)
            total_loss += loss.item()
            total_bce += bce.item()
            total_kld += kld.item()

    n = len(test_loader.dataset)
    return {
        "loss": total_loss / n,
        "bce": total_bce / n,
        "kld": total_kld / n,
    }
