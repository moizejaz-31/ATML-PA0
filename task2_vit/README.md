# Task 2 — Understanding Vision Transformer (ViT)

## Objectives & Key Findings

1. **Pretrained Classification:**
   - ViT-Base models process 224x224 input images broken into 14x14 patches (196 patch tokens + 1 CLS token).

2. **Patch Attention Maps:**
   - Extracting CLS token attention from the final self-attention layer reveals sharp focus on foreground semantic objects.
   - Unlike Grad-CAM (gradient-based), ViT CLS attention directly mirrors self-attention weights.

3. **Patch Masking Robustness:**
   - ViT displays high robustness to random patch masking (up to 30-50% missing patches).
   - Structured center masking causes much faster degradation due to loss of contiguous spatial context.

4. **CLS Token vs. Mean-Pooling Probe:**
   - Linear probing shows comparable accuracy between CLS token embeddings and mean-pooled patch embeddings.
