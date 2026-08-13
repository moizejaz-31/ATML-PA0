import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import umap

def plot_tsne(features, labels, save_path=None, title="t-SNE Feature Projection"):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='tab10', alpha=0.7, s=20)
    plt.colorbar(scatter, label='Class Label')
    plt.title(title)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.grid(True, linestyle='--', alpha=0.5)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def plot_umap(features, labels, save_path=None, title="UMAP Feature Projection"):
    reducer = umap.UMAP(n_components=2, random_state=42)
    embeddings = reducer.fit_transform(features)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='tab10', alpha=0.7, s=20)
    plt.colorbar(scatter, label='Class Label')
    plt.title(title)
    plt.xlabel("UMAP Dim 1")
    plt.ylabel("UMAP Dim 2")
    plt.grid(True, linestyle='--', alpha=0.5)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def plot_layer_tsne_individual(features, labels, layer_name, save_path=None):
    """
    Plots single-layer t-SNE scatter plot matching exact assignment style:
    Title: "t-SNE visualization of {layer_name} features"
    Legend: Discrete classes 0 through 9
    """
    tsne = TSNE(n_components=2, random_state=42)
    embeddings = tsne.fit_transform(features)

    plt.figure(figsize=(7, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for c in range(10):
        mask = (labels == c)
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], color=colors[c], label=str(c), alpha=0.7, s=20)

    plt.title(f"t-SNE visualization of {layer_name} features")
    plt.legend(title="Classes", loc='upper right', bbox_to_anchor=(1.0, 1.0))
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def plot_training_curves(df_dict, save_path=None, title="Training & Validation Performance"):
    """
    df_dict: dict mapping experiment name -> DataFrame with columns [epoch, train_loss, train_acc, val_loss, val_acc]
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for exp_name, df in df_dict.items():
        epochs = df['epoch']
        axes[0].plot(epochs, df['train_loss'], label=f'{exp_name} Train Loss', linestyle='--')
        axes[0].plot(epochs, df['val_loss'], label=f'{exp_name} Val Loss', linestyle='-')

        axes[1].plot(epochs, df['train_acc'], label=f'{exp_name} Train Acc', linestyle='--')
        axes[1].plot(epochs, df['val_acc'], label=f'{exp_name} Val Acc', linestyle='-')

    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend()

    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend()

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def plot_multi_layer_projections(activations_dict, labels, method='tsne', save_path=None):
    """
    activations_dict: dict mapping layer_name -> feature matrix (N, D)
    """
    num_layers = len(activations_dict)
    fig, axes = plt.subplots(1, num_layers, figsize=(6 * num_layers, 5))
    if num_layers == 1:
        axes = [axes]

    for idx, (layer_name, feats) in enumerate(activations_dict.items()):
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:
            reducer = umap.UMAP(n_components=2, random_state=42)
        
        emb = reducer.fit_transform(feats)
        scatter = axes[idx].scatter(emb[:, 0], emb[:, 1], c=labels, cmap='tab10', alpha=0.7, s=15)
        axes[idx].set_title(f"{layer_name.capitalize()} ({method.upper()})")
        axes[idx].grid(True, linestyle='--', alpha=0.5)

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(scatter, cax=cbar_ax, label='Class Label')
    fig.suptitle(f"Feature Hierarchy Evolution Across Depth ({method.upper()})", fontsize=14)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def plot_layer_similarity_matrix(activations_dict, save_path=None):
    """
    Computes and plots cross-layer representation cosine similarity matrix across layer depth.
    """
    layer_names = list(activations_dict.keys())
    num_layers = len(layer_names)
    sim_matrix = np.zeros((num_layers, num_layers))

    for i, name1 in enumerate(layer_names):
        feats1 = activations_dict[name1]
        for j, name2 in enumerate(layer_names):
            feats2 = activations_dict[name2]
            # Match feature dimensions via mean pooling or dot product if dimensions differ
            min_dim = min(feats1.shape[1], feats2.shape[1])
            f1 = feats1[:, :min_dim]
            f2 = feats2[:, :min_dim]
            # Compute average sample-wise cosine similarity
            norm1 = f1 / (np.linalg.norm(f1, axis=1, keepdims=True) + 1e-8)
            norm2 = f2 / (np.linalg.norm(f2, axis=1, keepdims=True) + 1e-8)
            sim_matrix[i, j] = np.mean(np.sum(norm1 * norm2, axis=1))

    plt.figure(figsize=(7, 6))
    sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="magma", xticklabels=layer_names, yticklabels=layer_names)
    plt.title("Cross-Layer Representation Cosine Similarity")
    plt.xlabel("Layer Stage")
    plt.ylabel("Layer Stage")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def plot_confusion_and_similarity(model, val_loader, class_names, device="cpu", save_path=None):
    """
    Evaluates model feature representations on val_loader using Nearest Class Centroid
    classification, plots normalized confusion matrix, and computes class centroid cosine similarity.
    """
    model.eval()
    all_targets = []
    all_feats = []

    def hook_fn(module, input, output):
        all_feats.append(output.detach().cpu())

    handle = model.avgpool.register_forward_hook(hook_fn)

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            if inputs.shape[-2:] != (224, 224):
                inputs = torch.nn.functional.interpolate(inputs, size=(224, 224), mode='bilinear', align_corners=False)
            _ = model(inputs)
            all_targets.extend(targets.numpy())

    handle.remove()

    all_targets = np.array(all_targets)
    all_feats = torch.cat(all_feats, dim=0).view(len(all_targets), -1).numpy()

    # Normalize feature vectors for cosine distance computation
    norms = np.linalg.norm(all_feats, axis=1, keepdims=True) + 1e-8
    norm_feats = all_feats / norms

    # Class centroids & intra-model similarity
    num_classes = len(class_names)
    centroids = np.zeros((num_classes, norm_feats.shape[1]))
    for c in range(num_classes):
        mask = (all_targets == c)
        if np.any(mask):
            centroids[c] = norm_feats[mask].mean(axis=0)
            centroids[c] /= (np.linalg.norm(centroids[c]) + 1e-8)

    # Nearest Centroid classification for accurate pre-trained feature confusion matrix
    sample_sims = norm_feats @ centroids.T  # (N, num_classes)
    all_preds = sample_sims.argmax(axis=1)

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sim_matrix = cosine_similarity(centroids)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("Normalized Confusion Matrix (Nearest Centroid)")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="viridis", xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title("Class Centroid Cosine Feature Similarity")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Class")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def plot_model_comparison(feats1, feats2, labels, title1="ResNet-152", title2="ResNet-18", method='tsne', save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if method.lower() == 'tsne':
        r1 = TSNE(n_components=2, random_state=42)
        r2 = TSNE(n_components=2, random_state=42)
    else:
        r1 = umap.UMAP(n_components=2, random_state=42)
        r2 = umap.UMAP(n_components=2, random_state=42)

    emb1 = r1.fit_transform(feats1)
    emb2 = r2.fit_transform(feats2)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for c in range(10):
        mask1 = (labels == c)
        axes[0].scatter(emb1[mask1, 0], emb1[mask1, 1], color=colors[c], label=str(c), alpha=0.7, s=20)
        mask2 = (labels == c)
        axes[1].scatter(emb2[mask2, 0], emb2[mask2, 1], color=colors[c], label=str(c), alpha=0.7, s=20)

    axes[0].set_title(f"{title1} ({method.upper()})")
    axes[0].grid(True, linestyle='--', alpha=0.3)
    axes[0].legend(title="Classes", loc='upper right', bbox_to_anchor=(1.0, 1.0))

    axes[1].set_title(f"{title2} ({method.upper()})")
    axes[1].grid(True, linestyle='--', alpha=0.3)
    axes[1].legend(title="Classes", loc='upper right', bbox_to_anchor=(1.0, 1.0))

    fig.suptitle(f"Latent Feature Space Manifold Comparison ({method.upper()})", fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def plot_cross_model_class_similarity(feats1, feats2, labels, class_names, title1="ResNet-152", title2="ResNet-18", save_path=None):
    """
    Computes intra-model class centroid similarity matrices for model 1 and model 2
    and plots them side-by-side for comparative representational similarity analysis.
    """
    num_classes = len(class_names)
    c1 = np.zeros((num_classes, feats1.shape[1]))
    c2 = np.zeros((num_classes, feats2.shape[1]))

    for c in range(num_classes):
        m1 = (labels == c)
        if np.any(m1):
            c1[c] = feats1[m1].mean(axis=0)
            c1[c] /= (np.linalg.norm(c1[c]) + 1e-8)
            c2[c] = feats2[m1].mean(axis=0)
            c2[c] /= (np.linalg.norm(c2[c]) + 1e-8)

    sim1 = cosine_similarity(c1)
    sim2 = cosine_similarity(c2)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    sns.heatmap(sim1, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f"{title1} Class Feature Cosine Similarity", fontsize=12)
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Class")
    axes[0].tick_params(axis='x', rotation=45)

    sns.heatmap(sim2, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title(f"{title2} Class Feature Cosine Similarity", fontsize=12)
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Class")
    axes[1].tick_params(axis='x', rotation=45)

    fig.suptitle(f"Cross-Model Representational Similarity Comparison ({title1} vs {title2})", fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def plot_cluster_metrics_comparison(feats1, feats2, labels, title1="ResNet-152", title2="ResNet-18", save_path=None):
    """
    Calculates and plots intra-class tightness (mean distance to centroid) and 
    inter-class separation (mean distance between centroids) for two models.
    """
    def calc_metrics(feats, labels):
        classes = np.unique(labels)
        centroids = []
        intra_dists = []
        for c in classes:
            mask = (labels == c)
            pts = feats[mask]
            cent = pts.mean(axis=0)
            centroids.append(cent)
            d = np.linalg.norm(pts - cent, axis=1).mean()
            intra_dists.append(d)

        centroids = np.array(centroids)
        inter_dists = []
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                inter_dists.append(np.linalg.norm(centroids[i] - centroids[j]))

        return np.mean(intra_dists), np.mean(inter_dists)

    intra1, inter1 = calc_metrics(feats1, labels)
    intra2, inter2 = calc_metrics(feats2, labels)

    ratio1 = inter1 / (intra1 + 1e-8)
    ratio2 = inter2 / (intra2 + 1e-8)

    metrics = ['Intra-Class Variance\n(Lower = Tighter)', 'Inter-Class Margin\n(Higher = More Separated)', 'Separability Ratio\n(Inter / Intra)']
    m1_vals = [intra1, inter1, ratio1]
    m2_vals = [intra2, inter2, ratio2]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(x - width / 2, m1_vals, width, label=title1, color='royalblue')
    rects2 = ax.bar(x + width / 2, m2_vals, width, label=title2, color='darkorange')

    ax.set_ylabel('Metric Value (Normalized Feature Space)')
    ax.set_title(f'Latent Feature Space Clustering Quantitative Comparison ({title1} vs {title2})', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in rects1 + rects2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def plot_misclassified_images(model, val_loader, class_names, device="cpu", num_images=10, save_path=None):
    """
    Finds and displays validation images misclassified with the highest model confidence.
    """
    model.eval()
    misclassified = []

    # ImageNet mean/std for accurate un-normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs_dev = inputs.to(device)
            if inputs_dev.shape[-2:] != (224, 224):
                inputs_dev = torch.nn.functional.interpolate(inputs_dev, size=(224, 224), mode='bilinear', align_corners=False)
            outputs = model(inputs_dev)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            for img, target, pred, conf in zip(inputs, targets, preds.cpu(), confs.cpu()):
                if pred != target:
                    misclassified.append({
                        'image': img.numpy().transpose(1, 2, 0),
                        'true_label': class_names[target.item()],
                        'pred_label': class_names[pred.item()],
                        'confidence': conf.item()
                    })

    # Sort by highest confidence in wrong prediction
    misclassified.sort(key=lambda x: x['confidence'], reverse=True)
    top_misclassified = misclassified[:num_images]

    cols = 5
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows))
    axes = axes.flatten() if num_images > 1 else [axes]

    for i in range(num_images):
        if i < len(top_misclassified):
            item = top_misclassified[i]
            img = item['image'] * std + mean
            img = np.clip(img, 0, 1)

            axes[i].imshow(img)
            axes[i].set_title(f"True: {item['true_label']}\nPred: {item['pred_label']} ({item['confidence']*100:.1f}%)",
                              fontsize=10, color='crimson')
        axes[i].axis('off')

    for i in range(len(top_misclassified), len(axes)):
        axes[i].axis('off')

    plt.suptitle("Top High-Confidence Misclassified Samples", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


