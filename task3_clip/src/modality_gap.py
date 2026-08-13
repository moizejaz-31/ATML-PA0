import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def compute_modality_gap_metrics(img_embeds, txt_embeds):
    """
    Compute quantitative metrics for the modality gap:
      - mean_gap:        L2 distance between image and text centroid
      - mean_cosine_sim: average cosine similarity between matched pairs
      - intra_img_dist:  mean pairwise L2 distance within image embeddings
      - intra_txt_dist:  mean pairwise L2 distance within text embeddings
    """
    img_centroid = img_embeds.mean(axis=0)
    txt_centroid = txt_embeds.mean(axis=0)
    mean_gap = np.linalg.norm(img_centroid - txt_centroid)

    # Cosine similarity between matched pairs
    img_n = img_embeds / (np.linalg.norm(img_embeds, axis=-1, keepdims=True) + 1e-8)
    txt_n = txt_embeds / (np.linalg.norm(txt_embeds, axis=-1, keepdims=True) + 1e-8)
    cosine_sims = np.sum(img_n * txt_n, axis=-1)
    mean_cosine_sim = cosine_sims.mean()

    # Intra-modal mean distances (sample from pairs for efficiency)
    n = len(img_embeds)
    rng = np.random.default_rng(42)
    pairs = rng.choice(n, size=(min(500, n * (n - 1) // 2), 2), replace=True)
    intra_img = np.mean([np.linalg.norm(img_embeds[i] - img_embeds[j])
                         for i, j in pairs if i != j]) if n > 1 else 0.0
    intra_txt = np.mean([np.linalg.norm(txt_embeds[i] - txt_embeds[j])
                         for i, j in pairs if i != j]) if n > 1 else 0.0

    return {
        "centroid_gap": mean_gap,
        "mean_cosine_sim": mean_cosine_sim,
        "intra_img_dist": intra_img,
        "intra_txt_dist": intra_txt,
    }


def plot_modality_gap(img_embeds, txt_embeds, labels, class_names=None,
                      title="CLIP Modality Gap Visualization", save_path=None):
    """
    t-SNE projection of image and text embeddings, colour-coded by modality.
    """
    n_samples = len(img_embeds)
    combined = np.vstack([img_embeds, txt_embeds])

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, n_samples // 3)))
    embeds_2d = tsne.fit_transform(combined)

    img_2d = embeds_2d[:n_samples]
    txt_2d = embeds_2d[n_samples:]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(img_2d[:, 0], img_2d[:, 1], c='#3498db', alpha=0.7,
               label='Image Embeddings', s=40, edgecolors='white', linewidth=0.5)
    ax.scatter(txt_2d[:, 0], txt_2d[:, 1], c='#e74c3c', alpha=0.7,
               label='Text Embeddings', s=40, marker='^', edgecolors='white', linewidth=0.5)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def plot_modality_gap_by_class(img_embeds, txt_embeds, labels, class_names,
                               title="CLIP Modality Gap — Class-Coloured",
                               save_path=None):
    """
    t-SNE projection with each class in a different colour, image=circle, text=triangle.
    """
    n_samples = len(img_embeds)
    combined = np.vstack([img_embeds, txt_embeds])

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, n_samples // 3)))
    embeds_2d = tsne.fit_transform(combined)

    img_2d = embeds_2d[:n_samples]
    txt_2d = embeds_2d[n_samples:]

    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap('tab10', len(unique_labels))

    fig, ax = plt.subplots(figsize=(12, 8))
    for li in unique_labels:
        mask = labels == li
        color = cmap(li)
        name = class_names[li] if class_names else str(li)
        ax.scatter(img_2d[mask, 0], img_2d[mask, 1], c=[color], alpha=0.7,
                   s=40, label=f'{name} (img)', marker='o', edgecolors='white', linewidth=0.3)
        ax.scatter(txt_2d[mask, 0], txt_2d[mask, 1], c=[color], alpha=0.9,
                   s=80, marker='^', edgecolors='black', linewidth=0.5)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.legend(fontsize=8, ncol=2, loc='best', framealpha=0.8)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
