import os
# pyrefly: ignore [missing-import]
import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_latent_space(model, test_loader, device="cpu", save_path=None,
                      title="VAE 2D Latent Space — MNIST Test Set"):
    """
    Plot encoder μ(x) for all test images, colour-coded by digit label.
    """
    model.eval()
    mus = []
    labels = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            mu, _ = model.encode(data.view(-1, 784))
            mus.append(mu.cpu().numpy())
            labels.append(target.numpy())

    mus = np.vstack(mus)
    labels = np.concatenate(labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(mus[:, 0], mus[:, 1], c=labels, cmap='tab10',
                         alpha=0.6, s=8, edgecolors='none')
    cbar = plt.colorbar(scatter, ax=ax, label='Digit Label')
    cbar.set_ticks(range(10))
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("$z_1$", fontsize=12)
    ax.set_ylabel("$z_2$", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

    return mus, labels


def plot_latent_space_with_centroids(model, test_loader, device="cpu", save_path=None):
    """
    Latent space scatter with per-digit centroid annotations.
    """
    mus, labels = plot_latent_space.__wrapped__(model, test_loader, device) \
        if hasattr(plot_latent_space, '__wrapped__') \
        else _extract_latent(model, test_loader, device)

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.get_cmap('tab10', 10)

    for digit in range(10):
        mask = labels == digit
        ax.scatter(mus[mask, 0], mus[mask, 1], c=[cmap(digit)],
                   alpha=0.4, s=6, label=f'{digit}')
        centroid = mus[mask].mean(axis=0)
        ax.scatter(centroid[0], centroid[1], c=[cmap(digit)],
                   s=200, marker='*', edgecolors='black', linewidth=0.8, zorder=5)
        ax.annotate(str(digit), centroid, fontsize=11, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    ax.set_title("2D Latent Space with Per-Digit Centroids (★)", fontsize=14, fontweight='bold')
    ax.set_xlabel("$z_1$", fontsize=12)
    ax.set_ylabel("$z_2$", fontsize=12)
    ax.legend(fontsize=8, ncol=2, loc='upper right', framealpha=0.8)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def _extract_latent(model, test_loader, device):
    model.eval()
    mus, labels = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            mu, _ = model.encode(data.view(-1, 784))
            mus.append(mu.cpu().numpy())
            labels.append(target.numpy())
    return np.vstack(mus), np.concatenate(labels)


def plot_reconstructions(model, test_loader, device="cpu", num_images=10,
                         save_path=None, title="VAE Reconstructions — Original vs. Decoded"):
    """
    Show original and reconstructed images side by side with per-pixel error.
    """
    model.eval()
    data, targets = next(iter(test_loader))
    data = data[:num_images].to(device)
    targets = targets[:num_images]

    with torch.no_grad():
        recon, _, _ = model(data)

    fig, axes = plt.subplots(3, num_images, figsize=(num_images * 1.6, 5))

    for i in range(num_images):
        orig = data[i].squeeze().cpu().numpy()
        rec = recon[i].view(28, 28).cpu().numpy()
        err = np.abs(orig - rec)

        axes[0, i].imshow(orig, cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel("Original", fontsize=11, fontweight='bold')

        axes[1, i].imshow(rec, cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel("Reconstructed", fontsize=11, fontweight='bold')

        axes[2, i].imshow(err, cmap='hot')
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel("Error", fontsize=11, fontweight='bold')

        axes[0, i].set_title(f'Label: {targets[i].item()}', fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def plot_per_digit_reconstructions(model, test_loader, device="cpu", save_path=None):
    """
    Show one reconstruction per digit class (0–9).
    """
    model.eval()
    digit_examples = {}
    for data, targets in test_loader:
        for img, label in zip(data, targets):
            l = label.item()
            if l not in digit_examples:
                digit_examples[l] = img
            if len(digit_examples) == 10:
                break
        if len(digit_examples) == 10:
            break

    fig, axes = plt.subplots(2, 10, figsize=(16, 3.5))
    with torch.no_grad():
        for digit in range(10):
            img = digit_examples[digit].unsqueeze(0).to(device)
            recon, _, _ = model(img)

            axes[0, digit].imshow(img.squeeze().cpu().numpy(), cmap='gray')
            axes[0, digit].axis('off')
            axes[0, digit].set_title(str(digit), fontsize=10, fontweight='bold')

            axes[1, digit].imshow(recon.view(28, 28).cpu().numpy(), cmap='gray')
            axes[1, digit].axis('off')

    axes[0, 0].set_ylabel("Original", fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel("Recon.", fontsize=10, fontweight='bold')
    fig.suptitle("Per-Digit Reconstruction Quality (0–9)", fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def sample_prior(model, num_samples=25, latent_dim=2, device="cpu", save_path=None,
                 title="Generated Digits from Prior $z \\sim \\mathcal{N}(0, I)$"):
    """
    Sample z from N(0, I) and decode to generate new digit images.
    """
    model.eval()
    z = torch.randn(num_samples, latent_dim).to(device)

    with torch.no_grad():
        samples = model.decode(z).view(-1, 28, 28).cpu().numpy()

    grid_size = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(6, 6))
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < num_samples:
                axes[i, j].imshow(samples[idx], cmap='gray')
            axes[i, j].axis('off')

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def plot_latent_manifold_grid(model, n=20, digit_size=28, device="cpu",
                               save_path=None,
                               title="2D Latent Manifold Grid — Continuous Digit Transitions"):
    """
    Sample z1, z2 on a grid using inverse CDF of N(0,1) and decode to visualize
    the full 2D latent manifold.
    """
    model.eval()
    from scipy.stats import norm
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    figure = np.zeros((digit_size * n, digit_size * n))

    with torch.no_grad():
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = torch.tensor([[xi, yi]], dtype=torch.float32).to(device)
                x_decoded = model.decode(z_sample).view(digit_size, digit_size).cpu().numpy()
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = x_decoded

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(figure, cmap='gray')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def plot_latent_interpolation(model, test_loader, device="cpu", num_steps=10,
                               save_path=None):
    """
    Interpolate linearly between two random test images in latent space
    and decode the intermediate points.
    """
    model.eval()
    data, targets = next(iter(test_loader))
    # Pick two images of different digits
    idx1, idx2 = 0, 1
    for i in range(len(targets)):
        if targets[i] != targets[0]:
            idx2 = i
            break

    img1 = data[idx1:idx1+1].to(device)
    img2 = data[idx2:idx2+1].to(device)

    with torch.no_grad():
        mu1, _ = model.encode(img1.view(-1, 784))
        mu2, _ = model.encode(img2.view(-1, 784))

        alphas = np.linspace(0, 1, num_steps)
        interpolated = []
        for alpha in alphas:
            z_interp = (1 - alpha) * mu1 + alpha * mu2
            decoded = model.decode(z_interp).view(28, 28).cpu().numpy()
            interpolated.append(decoded)

    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 1.5, 2))
    for i, img in enumerate(interpolated):
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        if i == 0:
            axes[i].set_title(f'{targets[idx1].item()}', fontsize=10, fontweight='bold', color='#2980b9')
        elif i == num_steps - 1:
            axes[i].set_title(f'{targets[idx2].item()}', fontsize=10, fontweight='bold', color='#c0392b')
        else:
            axes[i].set_title(f'α={alphas[i]:.1f}', fontsize=8)

    fig.suptitle(f'Latent Space Interpolation: Digit {targets[idx1].item()} → {targets[idx2].item()}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def compute_reconstruction_mse_per_digit(model, test_loader, device="cpu"):
    """
    Compute mean reconstruction MSE per digit class.
    Returns dict mapping digit -> mean MSE.
    """
    model.eval()
    digit_errors = {d: [] for d in range(10)}

    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            recon, _, _ = model(data)
            recon = recon.view(-1, 784)
            data_flat = data.view(-1, 784)
            mse = ((recon.cpu() - data_flat.cpu()) ** 2).mean(dim=1).numpy()
            for err, label in zip(mse, targets.numpy()):
                digit_errors[label].append(err)

    return {d: float(np.mean(errs)) for d, errs in digit_errors.items()}
