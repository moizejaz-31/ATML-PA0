import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from scipy.stats import entropy


def extract_cls_attention(model, pixel_values):
    """
    Run model with output_attentions=True, average attention over heads for final layer,
    and extract 14x14 CLS-to-patch attention map.
    """
    with torch.no_grad():
        outputs = model(pixel_values, output_attentions=True)

    # outputs.attentions is a tuple of (num_layers,) with shape (batch, heads, seq_len, seq_len)
    last_layer_att = outputs.attentions[-1]  # shape: (1, num_heads, 197, 197)
    mean_att = last_layer_att.mean(dim=1).squeeze(0)  # shape: (197, 197)

    # Extract CLS token attending to patch tokens (index 0 -> indices 1..196)
    cls_att = mean_att[0, 1:].cpu().numpy()  # 196 elements
    cls_att_grid = cls_att.reshape(14, 14)

    # Normalize to [0, 1]
    cls_att_grid = (cls_att_grid - cls_att_grid.min()) / (cls_att_grid.max() - cls_att_grid.min() + 1e-8)
    return cls_att_grid, outputs.logits


def extract_attention_rollout(model, pixel_values):
    """
    Compute Attention Rollout (Abnar & Zuidema, 2020) across all transformer layers.
    Accounts for residual connections V_l = 0.5 * A_l + 0.5 * I to produce crisp, sharp saliency maps.
    """
    with torch.no_grad():
        outputs = model(pixel_values, output_attentions=True)

    seq_len = outputs.attentions[0].shape[-1]
    joint_att = torch.eye(seq_len)

    for att in outputs.attentions:
        a = att.squeeze(0).mean(dim=0).cpu()  # average over heads
        a_reg = 0.5 * a + 0.5 * torch.eye(seq_len)  # add residual connection
        a_reg = a_reg / a_reg.sum(dim=-1, keepdim=True)  # re-normalize rows
        joint_att = torch.matmul(a_reg, joint_att)

    rollout_cls = joint_att[0, 1:].numpy().reshape(14, 14)
    rollout_cls = (rollout_cls - rollout_cls.min()) / (rollout_cls.max() - rollout_cls.min() + 1e-8)
    return rollout_cls, outputs.logits


def extract_per_head_attention(model, pixel_values, layer_idx=-1):
    """
    Extract per-head CLS-to-patch attention maps from a specific layer.
    Returns: numpy array of shape (num_heads, 14, 14) and logits.
    """
    with torch.no_grad():
        outputs = model(pixel_values, output_attentions=True)

    layer_att = outputs.attentions[layer_idx]  # (1, num_heads, 197, 197)
    layer_att = layer_att.squeeze(0)  # (num_heads, 197, 197)

    # CLS token row (index 0), patch columns (indices 1..196)
    cls_att_heads = layer_att[:, 0, 1:].cpu().numpy()  # (num_heads, 196)

    head_grids = []
    for h in range(cls_att_heads.shape[0]):
        grid = cls_att_heads[h].reshape(14, 14)
        grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
        head_grids.append(grid)

    return np.stack(head_grids), outputs.logits


def extract_attention_across_layers(model, pixel_values, layer_indices=None):
    """
    Extract head-averaged CLS attention maps from multiple layers.
    Returns dict mapping layer_index -> (14, 14) attention grid.
    """
    with torch.no_grad():
        outputs = model(pixel_values, output_attentions=True)

    if layer_indices is None:
        num_layers = len(outputs.attentions)
        layer_indices = [0, num_layers // 4, num_layers // 2, num_layers - 1]

    layer_maps = {}
    for li in layer_indices:
        att = outputs.attentions[li]  # (1, heads, 197, 197)
        mean_att = att.mean(dim=1).squeeze(0)  # (197, 197)
        cls_att = mean_att[0, 1:].cpu().numpy().reshape(14, 14)
        cls_att = (cls_att - cls_att.min()) / (cls_att.max() - cls_att.min() + 1e-8)
        layer_maps[li] = cls_att

    return layer_maps


def overlay_attention_map(image, att_grid, save_path=None, title="ViT CLS Attention Overlay"):
    """
    Upsample 14x14 attention grid and overlay semi-transparent red heatmap on original PIL image.
    """
    img_np = np.array(image.convert("RGB"))
    h, w, _ = img_np.shape

    # Resize attention grid to original image size
    att_pil = Image.fromarray((att_grid * 255).astype(np.uint8)).resize((w, h), resample=Image.BICUBIC)
    att_map = np.array(att_pil) / 255.0

    plt.figure(figsize=(7, 7))
    plt.imshow(img_np)
    plt.imshow(att_map, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.title(title, fontsize=13)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def plot_attention_tripanel(image, att_grid, title="", save_path=None):
    """
    Three-panel figure: Original Image | Raw 14x14 Attention Map | Overlay on Image.
    """
    img_np = np.array(image.convert("RGB"))
    h, w, _ = img_np.shape

    att_pil = Image.fromarray((att_grid * 255).astype(np.uint8)).resize((w, h), resample=Image.BICUBIC)
    att_up = np.array(att_pil) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(img_np)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis('off')

    im = axes[1].imshow(att_grid, cmap='inferno', interpolation='nearest')
    axes[1].set_title("CLS Attention (14×14)", fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(img_np)
    axes[2].imshow(att_up, cmap='jet', alpha=0.5)
    axes[2].set_title("Attention Overlay", fontsize=12)
    axes[2].axis('off')

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def plot_attention_methods_comparison(image, att_last_grid, att_rollout_grid, title="", save_path=None):
    """
    Four-panel figure: Original Image | Raw Last-Layer Mean | Attention Rollout | Rollout Heatmap Overlay.
    Compares naive final layer mean against Attention Rollout (Abnar & Zuidema 2020).
    """
    img_np = np.array(image.convert("RGB"))
    h, w, _ = img_np.shape

    att_pil = Image.fromarray((att_rollout_grid * 255).astype(np.uint8)).resize((w, h), resample=Image.BICUBIC)
    att_up = np.array(att_pil) / 255.0

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(img_np)
    axes[0].set_title("Original Input Image", fontsize=12)
    axes[0].axis('off')

    im1 = axes[1].imshow(att_last_grid, cmap='inferno', interpolation='nearest')
    axes[1].set_title("Raw Final Layer Mean", fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(att_rollout_grid, cmap='inferno', interpolation='nearest')
    axes[2].set_title("Attention Rollout (All Layers)", fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(img_np)
    axes[3].imshow(att_up, cmap='jet', alpha=0.5)
    axes[3].set_title("Attention Rollout Overlay", fontsize=12)
    axes[3].axis('off')

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def plot_per_head_attention(image, head_grids, save_path=None, title="Per-Head CLS Attention — Final Layer"):
    """
    Plot a grid showing each attention head's CLS-to-patch map overlaid on the image.
    head_grids: numpy array (num_heads, 14, 14).
    """
    img_np = np.array(image.convert("RGB"))
    h, w, _ = img_np.shape
    num_heads = head_grids.shape[0]

    cols = 4
    rows = (num_heads + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for i in range(num_heads):
        att_pil = Image.fromarray((head_grids[i] * 255).astype(np.uint8)).resize((w, h), resample=Image.BICUBIC)
        att_up = np.array(att_pil) / 255.0

        axes[i].imshow(img_np)
        axes[i].imshow(att_up, cmap='jet', alpha=0.5)
        axes[i].set_title(f"Head {i}", fontsize=11)
        axes[i].axis('off')

    # Hide unused axes
    for i in range(num_heads, len(axes)):
        axes[i].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def plot_attention_across_layers(image, layer_maps, save_path=None, title="CLS Attention Evolution Across Transformer Layers"):
    """
    Side-by-side attention overlays from multiple layers.
    layer_maps: dict mapping layer_index -> (14, 14) grid.
    """
    img_np = np.array(image.convert("RGB"))
    h, w, _ = img_np.shape

    sorted_layers = sorted(layer_maps.keys())
    n = len(sorted_layers)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, li in zip(axes, sorted_layers):
        att_pil = Image.fromarray((layer_maps[li] * 255).astype(np.uint8)).resize((w, h), resample=Image.BICUBIC)
        att_up = np.array(att_pil) / 255.0

        ax.imshow(img_np)
        ax.imshow(att_up, cmap='jet', alpha=0.5)
        ax.set_title(f"Layer {li + 1}", fontsize=12)
        ax.axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def plot_attention_entropy(model, pixel_values, save_path=None, title="Per-Head Attention Entropy — Final Layer"):
    """
    Compute and plot Shannon entropy of each head's CLS-to-patch attention distribution.
    High entropy = diffuse; Low entropy = sharply focused.
    """
    with torch.no_grad():
        outputs = model(pixel_values, output_attentions=True)

    last_att = outputs.attentions[-1].squeeze(0)  # (num_heads, 197, 197)
    num_heads = last_att.shape[0]

    entropies = []
    for h in range(num_heads):
        # CLS token attention distribution over patches
        cls_dist = last_att[h, 0, 1:].cpu().numpy()  # 196 values
        cls_dist = cls_dist / (cls_dist.sum() + 1e-8)  # normalize to prob distribution
        ent = entropy(cls_dist, base=2)
        entropies.append(ent)

    max_entropy = np.log2(196)  # uniform distribution entropy

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, num_heads))
    bars = ax.bar(range(num_heads), entropies, color=colors)
    ax.axhline(y=max_entropy, color='red', linestyle='--', alpha=0.7, label=f'Max Entropy (uniform) = {max_entropy:.2f} bits')
    ax.set_xlabel("Attention Head", fontsize=12)
    ax.set_ylabel("Shannon Entropy (bits)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(range(num_heads))
    ax.set_xticklabels([f"H{i}" for i in range(num_heads)])
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, ent in zip(bars, entropies):
        ax.annotate(f'{ent:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, ent),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def plot_top_bottom_patches(image, att_grid, top_k=10, bottom_k=10, save_path=None,
                            title="Top & Bottom Attended Patches"):
    """
    Highlight top-attended (green) and bottom-attended (red) patches on the image.
    """
    img_np = np.array(image.convert("RGB")).copy()
    h, w, _ = img_np.shape
    patch_h = h // 14
    patch_w = w // 14

    flat = att_grid.flatten()
    top_indices = np.argsort(flat)[-top_k:]
    bottom_indices = np.argsort(flat)[:bottom_k]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img_np)

    for idx in top_indices:
        r, c = divmod(idx, 14)
        rect = mpatches.Rectangle((c * patch_w, r * patch_h), patch_w, patch_h,
                                   linewidth=2, edgecolor='lime', facecolor='lime', alpha=0.3)
        ax.add_patch(rect)

    for idx in bottom_indices:
        r, c = divmod(idx, 14)
        rect = mpatches.Rectangle((c * patch_w, r * patch_h), patch_w, patch_h,
                                   linewidth=2, edgecolor='red', facecolor='red', alpha=0.25)
        ax.add_patch(rect)

    green_patch = mpatches.Patch(color='lime', alpha=0.5, label=f'Top-{top_k} Attended')
    red_patch = mpatches.Patch(color='red', alpha=0.4, label=f'Bottom-{bottom_k} Attended')
    ax.legend(handles=[green_patch, red_patch], loc='upper left', fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.axis('off')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
