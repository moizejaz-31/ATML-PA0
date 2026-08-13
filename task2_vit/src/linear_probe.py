import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset

def extract_vit_features_cifar10(model, processor, data_dir=None, num_samples=1000, batch_size=32, device="cpu"):
    """
    Extract CLS token embeddings and mean-pooled patch token embeddings from CIFAR-10 dataset.
    Locates existing extracted CIFAR-10 dataset (e.g. task1_resnet152/notebooks/data) without downloading.
    """
    base_workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates = [
        os.path.join(base_workspace, "task1_resnet152", "notebooks", "data"),
        os.path.join(base_workspace, "task1_resnet152", "data"),
        os.path.join(base_workspace, "data"),
        os.path.join(base_workspace, "task2_vit", "data"),
        os.path.abspath("./data"),
        os.path.abspath("../data"),
    ]
    if data_dir is not None:
        candidates.insert(0, data_dir)

    found_dir = candidates[0]
    for cand in candidates:
        if os.path.exists(os.path.join(cand, "cifar-10-batches-py")):
            found_dir = cand
            print(f"Located existing CIFAR-10 dataset at: {found_dir}")
            break
    data_dir = found_dir

    model.to(device)
    model.eval()

    try:
        cifar_dataset = CIFAR10(root=data_dir, train=False, download=False)
    except Exception:
        cifar_dataset = CIFAR10(root=data_dir, train=False, download=True)

    if num_samples is not None and num_samples < len(cifar_dataset):
        indices = np.random.choice(len(cifar_dataset), num_samples, replace=False)
        cifar_dataset = Subset(cifar_dataset, indices)

    cls_features = []
    mean_features = []
    labels_list = []

    print(f"Extracting ViT features for {len(cifar_dataset)} CIFAR-10 images...")

    for i in range(0, len(cifar_dataset), batch_size):
        batch_items = [cifar_dataset[j] for j in range(i, min(i + batch_size, len(cifar_dataset)))]
        images = [img for img, _ in batch_items]
        labels = [lbl for _, lbl in batch_items]

        inputs = processor(images=images, return_tensors="pt")['pixel_values'].to(device)

        with torch.no_grad():
            outputs = model.vit(inputs, output_hidden_states=True)
            last_hidden_state = outputs.last_hidden_state  # (batch, 197, 768)

            cls_emb = last_hidden_state[:, 0, :].cpu().numpy()  # CLS token (index 0)
            mean_emb = last_hidden_state[:, 1:, :].mean(dim=1).cpu().numpy()  # Mean of patch tokens (1..196)

            cls_features.append(cls_emb)
            mean_features.append(mean_emb)
            labels_list.extend(labels)

    X_cls = np.vstack(cls_features)
    X_mean = np.vstack(mean_features)
    y = np.array(labels_list)

    print(f"Extraction complete! Shapes: X_cls={X_cls.shape}, X_mean={X_mean.shape}, y={y.shape}")
    return X_cls, X_mean, y

def train_and_evaluate_probes(X_cls, X_mean, y, test_size=0.3, random_state=42):
    """
    Train Logistic Regression probes on CLS token vs Mean-pooled features.
    Returns dictionary with evaluation metrics and confusion matrices.
    """
    # Split CLS features
    X_cls_tr, X_cls_val, y_tr, y_val = train_test_split(X_cls, y, test_size=test_size, random_state=random_state, stratify=y)
    # Split Mean-pooled features using same indices
    X_mean_tr, X_mean_val, _, _ = train_test_split(X_mean, y, test_size=test_size, random_state=random_state, stratify=y)

    print("Training Logistic Regression on CLS token features...")
    clf_cls = LogisticRegression(max_iter=1000, random_state=random_state)
    clf_cls.fit(X_cls_tr, y_tr)
    y_pred_cls = clf_cls.predict(X_cls_val)
    acc_cls = accuracy_score(y_val, y_pred_cls)

    print("Training Logistic Regression on Mean-Pooled patch features...")
    clf_mean = LogisticRegression(max_iter=1000, random_state=random_state)
    clf_mean.fit(X_mean_tr, y_tr)
    y_pred_mean = clf_mean.predict(X_mean_val)
    acc_mean = accuracy_score(y_val, y_pred_mean)

    cm_cls = confusion_matrix(y_val, y_pred_cls)
    cm_mean = confusion_matrix(y_val, y_pred_mean)

    print(f"Probe Results -> CLS Token Accuracy: {acc_cls*100:.2f}% | Mean-Pooled Accuracy: {acc_mean*100:.2f}%")

    return {
        'acc_cls': acc_cls,
        'acc_mean': acc_mean,
        'cm_cls': cm_cls,
        'cm_mean': cm_mean,
        'clf_cls': clf_cls,
        'clf_mean': clf_mean,
        'y_val': y_val,
        'y_pred_cls': y_pred_cls,
        'y_pred_mean': y_pred_mean
    }

def plot_probe_comparison(results, save_path=None, title="CLS Token vs. Mean-Pooled Patch Probe Accuracy"):
    """
    Plot bar chart comparing linear probe validation accuracies.
    """
    acc_cls = results['acc_cls'] * 100
    acc_mean = results['acc_mean'] * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(['CLS Token Embedding', 'Mean-Pooled Patch Embeddings'], [acc_cls, acc_mean],
                  color=['#2980b9', '#16a085'], width=0.5)

    ax.set_ylabel('Linear Probe Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(acc_cls, acc_mean) + 15)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def plot_probe_confusion_matrices(results, class_names=None, save_path=None):
    """
    Plot side-by-side confusion matrices for CLS token probe vs Mean-pooled probe.
    """
    if class_names is None:
        class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(results['cm_cls'], annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title(f"CLS Token Probe (Acc: {results['acc_cls']*100:.1f}%)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=11)
    axes[0].set_ylabel('True Label', fontsize=11)

    sns.heatmap(results['cm_mean'], annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names)
    axes[1].set_title(f"Mean-Pooled Probe (Acc: {results['acc_mean']*100:.1f}%)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=11)
    axes[1].set_ylabel('True Label', fontsize=11)

    fig.suptitle('Linear Probe Confusion Matrices on CIFAR-10', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
