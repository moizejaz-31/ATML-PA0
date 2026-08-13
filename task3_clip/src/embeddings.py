import torch
import numpy as np


def extract_clip_embeddings(model, tokenizer, preprocess, dataset, class_names,
                            num_samples=100, device="cpu"):
    """
    Extract image embeddings and corresponding per-sample class text embeddings.

    Returns dict with keys:
      - raw_img:  (num_samples, embed_dim) raw image embeddings
      - raw_txt:  (num_samples, embed_dim) matched raw text embeddings
      - norm_img: (num_samples, embed_dim) L2-normalised image embeddings
      - norm_txt: (num_samples, embed_dim) matched L2-normalised text embeddings
      - text_class_raw:  (num_classes, embed_dim) per-class raw text embeddings
      - text_class_norm: (num_classes, embed_dim) per-class normalised text embeddings
      - labels:   (num_samples,) integer class labels
      - class_names: list of class name strings
    """
    # Sub-sample the dataset
    indices = list(range(min(num_samples, len(dataset))))
    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=num_samples, shuffle=False)
    images, labels = next(iter(loader))
    images = images.to(device)

    # Encode class prompts
    prompts = [f"a photo of a {c}." for c in class_names]
    text_tokens = tokenizer(prompts).to(device)

    with torch.no_grad():
        raw_img_embeds = model.encode_image(images).cpu().float().numpy()
        raw_txt_class = model.encode_text(text_tokens).cpu().float().numpy()

    # L2 normalisation
    norm_img_embeds = raw_img_embeds / np.linalg.norm(raw_img_embeds, axis=-1, keepdims=True)
    norm_txt_class = raw_txt_class / np.linalg.norm(raw_txt_class, axis=-1, keepdims=True)

    # Expand text embeddings to match per-sample image labels
    labs = labels.numpy()
    matched_raw_txt = raw_txt_class[labs]
    matched_norm_txt = norm_txt_class[labs]

    return {
        "raw_img": raw_img_embeds,
        "raw_txt": matched_raw_txt,
        "norm_img": norm_img_embeds,
        "norm_txt": matched_norm_txt,
        "text_class_raw": raw_txt_class,
        "text_class_norm": norm_txt_class,
        "labels": labs,
        "class_names": class_names,
    }
