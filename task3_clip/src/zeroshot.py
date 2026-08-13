import os
import torch
import numpy as np
import open_clip
from PIL import Image
from torch.utils.data import Dataset
import torchvision.datasets as datasets


class STL10FastDataset(Dataset):
    """
    STL-10 Test Dataset backed by pre-cached fast HuggingFace mirror (8,000 images).
    Falls back to torchvision STL10 if needed.
    """
    def __init__(self, root='../data', split='test', transform=None, download=True):
        self.transform = transform
        self.classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
        self.class_names = self.classes

        # Look for the fast cached pt file
        possible_paths = [
            os.path.join(root, 'stl10_test_fast.pt'),
            os.path.join(os.path.abspath(root), 'stl10_test_fast.pt'),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'data', 'stl10_test_fast.pt'),
            os.path.abspath('./data/stl10_test_fast.pt'),
            os.path.abspath('../data/stl10_test_fast.pt'),
        ]
        found = None
        for p in possible_paths:
            if os.path.exists(p):
                found = p
                break

        if found:
            cached = torch.load(found, weights_only=False)
            self.images = cached['images']
            self.labels = cached['labels']
            self.data = self.images
        else:
            tv_ds = datasets.STL10(root=root, split=split, download=download)
            self.data = tv_ds.data
            self.labels = tv_ds.labels
            self.images = self.data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_arr = self.images[idx]
        if isinstance(img_arr, np.ndarray):
            if img_arr.ndim == 3 and img_arr.shape[0] == 3:
                img_arr = np.transpose(img_arr, (1, 2, 0))
            img = Image.fromarray(img_arr)
        elif isinstance(img_arr, Image.Image):
            img = img_arr
        else:
            img = Image.fromarray(np.array(img_arr))

        target = int(self.labels[idx])
        if self.transform:
            img = self.transform(img)
        return img, target


def load_stl10_dataset(root='../data', split='test', transform=None, download=True):
    """
    Convenience loader that returns STL10 dataset instantly from fast cached mirror.
    """
    return STL10FastDataset(root=root, split=split, transform=transform, download=download)


def load_clip_model(model_name="ViT-B-32", pretrained="openai", device="cpu"):
    """
    Load an open_clip CLIP model, its preprocessing transforms, and tokenizer.
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer


def build_prompts(class_names, strategy="template"):
    """
    Build text prompts for zero-shot classification.

    Strategies:
      - 'plain':       bare class name, e.g. "cat"
      - 'template':    "a photo of a cat."
      - 'descriptive': richer context with category hints
      - 'ensemble':    multiple templates averaged (returns list-of-lists)
    """
    if strategy == "plain":
        return [f"{c}" for c in class_names]
    elif strategy == "template":
        return [f"a photo of a {c}." if c not in ('airplane',) else f"a photo of an {c}." for c in class_names]
    elif strategy == "descriptive":
        return [
            f"a centered, high-resolution photograph of a {c}, captured in natural lighting."
            if c not in ('airplane',) else
            f"a centered, high-resolution photograph of an {c}, captured in natural lighting."
            for c in class_names
        ]
    elif strategy == "ensemble":
        templates = [
            "a photo of a {}.",
            "a blurry photo of a {}.",
            "a close-up photo of a {}.",
            "a bright photo of a {}.",
            "a rendering of a {}.",
            "a photo of the large {}.",
            "a photo of the small {}.",
        ]
        return [[t.format(c) for t in templates] for c in class_names]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def encode_text_features(model, tokenizer, class_names, strategy="template", device="cpu"):
    """
    Encode class prompts into L2-normalised text feature vectors.
    Returns tensor of shape (num_classes, embed_dim).
    """
    prompts = build_prompts(class_names, strategy=strategy)

    with torch.no_grad():
        if strategy == "ensemble":
            class_features = []
            for class_prompts in prompts:
                tokens = tokenizer(class_prompts).to(device)
                feats = model.encode_text(tokens)
                feats /= feats.norm(dim=-1, keepdim=True)
                class_features.append(feats.mean(dim=0))
            text_features = torch.stack(class_features)
        else:
            tokens = tokenizer(prompts).to(device)
            text_features = model.encode_text(tokens)

        text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features


def evaluate_zeroshot(model, preprocess, tokenizer, dataset, class_names,
                      prompt_strategy="template", device="cpu", return_details=False):
    """
    Evaluate CLIP zero-shot accuracy on a dataset.
    """
    text_features = encode_text_features(
        model, tokenizer, class_names, strategy=prompt_strategy, device=device
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_sims = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            predictions = similarity.argmax(dim=-1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_preds.append(predictions.cpu())
            all_labels.append(labels.cpu())
            all_sims.append(similarity.cpu())

    accuracy = correct / total

    if return_details:
        return accuracy, {
            'predictions': torch.cat(all_preds).numpy(),
            'labels': torch.cat(all_labels).numpy(),
            'similarities': torch.cat(all_sims).numpy(),
        }
    return accuracy


def evaluate_zeroshot_aligned(model, preprocess, tokenizer, dataset, class_names,
                              R, prompt_strategy="template", device="cpu"):
    """
    Evaluate zero-shot accuracy after applying Procrustes rotation R to image features.
    """
    text_features = encode_text_features(
        model, tokenizer, class_names, strategy=prompt_strategy, device=device
    )

    R_tensor = torch.from_numpy(R).float().to(device)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            image_features = image_features @ R_tensor
            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            predictions = similarity.argmax(dim=-1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total
