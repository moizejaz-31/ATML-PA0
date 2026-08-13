import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def mask_patches(pixel_values, mask_ratio=0.3, mode="random"):
    """
    Mask patch regions (in 14x14 patch space) either randomly or in a structured center block.
    pixel_values shape: (1, 3, 224, 224) -> 14x14 patches of size 16x16.
    """
    masked = pixel_values.clone()
    num_patches_side = 14
    patch_size = 16
    total_patches = num_patches_side * num_patches_side

    num_to_mask = int(total_patches * mask_ratio)
    if num_to_mask == 0:
        return masked

    if mode == "random":
        mask_indices = np.random.choice(total_patches, num_to_mask, replace=False)
    elif mode == "structured":
        # Center crop masking in 14x14 grid
        side = int(np.ceil(np.sqrt(num_to_mask)))
        start = (num_patches_side - side) // 2
        mask_indices = []
        for r in range(start, start + side):
            for c in range(start, start + side):
                if len(mask_indices) < num_to_mask:
                    mask_indices.append(r * num_patches_side + c)
    else:
        raise ValueError("Mode must be 'random' or 'structured'")

    for idx in mask_indices:
        row = idx // num_patches_side
        col = idx % num_patches_side
        masked[:, :, row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size] = 0.0

    return masked

def unpreprocess_tensor(pixel_values):
    """
    Convert normalized pixel_values tensor back to PIL Image / RGB numpy array for display.
    """
    pv = pixel_values.squeeze(0).cpu().numpy()  # (3, 224, 224)
    # ImageNet mean & std
    mean = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)
    std = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)
    img_np = (pv * std + mean).clip(0, 1).transpose(1, 2, 0)
    return (img_np * 255).astype(np.uint8)

def plot_masked_grid(image, processor, mask_ratios=[0.0, 0.2, 0.5, 0.7], save_path=None, title="Patch Masking Modes & Ratios"):
    """
    Visual grid showing original image under random vs structured (center) patch masking across ratios.
    """
    from task2_vit.src.load_vit import preprocess_image
    pixel_values = preprocess_image(processor, image)

    rows = 2  # Random vs Structured
    cols = len(mask_ratios)

    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 6.5))
    modes = ["random", "structured"]
    mode_titles = ["Random Masking", "Structured (Center) Masking"]

    for r_idx, mode in enumerate(modes):
        for c_idx, ratio in enumerate(mask_ratios):
            masked_pv = mask_patches(pixel_values, mask_ratio=ratio, mode=mode)
            img_vis = unpreprocess_tensor(masked_pv)

            ax = axes[r_idx, c_idx]
            ax.imshow(img_vis)
            if r_idx == 0:
                ax.set_title(f'Mask Ratio: {int(ratio*100)}%', fontsize=11, fontweight='bold')
            if c_idx == 0:
                ax.set_ylabel(mode_titles[r_idx], fontsize=12, fontweight='bold')
            ax.axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def evaluate_patch_masking_multi(model, processor, images, mask_ratios=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9], modes=["random", "structured"]):
    """
    Evaluates probability degradation and top-1 accuracy drop across multiple images.
    Returns dict: {mode: {'probs': list_of_avg_probs, 'accs': list_of_top1_accs, 'per_image': {desc: list_of_probs}}}
    """
    from task2_vit.src.load_vit import preprocess_image

    results = {mode: {'probs': [], 'accs': [], 'per_image': {desc: [] for _, desc in images}} for mode in modes}

    for mode in modes:
        for ratio in mask_ratios:
            ratio_probs = []
            top1_correct = 0

            for img, desc in images:
                pixel_values = preprocess_image(processor, img).to(next(model.parameters()).device)

                # Get clean baseline target label
                with torch.no_grad():
                    clean_logits = model(pixel_values).logits
                    target_label = clean_logits.argmax(dim=-1).item()

                masked_pv = mask_patches(pixel_values, mask_ratio=ratio, mode=mode)
                with torch.no_grad():
                    logits = model(masked_pv).logits
                    probs = torch.softmax(logits, dim=-1)
                    target_prob = probs[0, target_label].item()
                    pred_label = logits.argmax(dim=-1).item()

                ratio_probs.append(target_prob)
                results[mode]['per_image'][desc].append(target_prob)
                if pred_label == target_label:
                    top1_correct += 1

            avg_prob = float(np.mean(ratio_probs))
            acc = top1_correct / len(images)
            results[mode]['probs'].append(avg_prob)
            results[mode]['accs'].append(acc)

    return results
