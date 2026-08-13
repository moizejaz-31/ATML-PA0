# Task 3 — CLIP Representation & Modality Gap

## Objectives & Key Findings

1. **Zero-Shot Prompt Strategies:**
   - Evaluated plain (`"cat"`), template (`"a photo of a cat."`), and descriptive (`"a high quality photo of a cat..."`) prompt strategies on STL-10.
   - Prompt engineering with template context significantly boosts zero-shot classification accuracy.

2. **Modality Gap Analysis:**
   - Raw image and text embeddings occupy distinct, non-overlapping regions in embedding space.
   - L2 normalization preserves contrastive similarity metrics while keeping modalities separated.

3. **Orthogonal Procrustes Alignment:**
   - Computing orthogonal transformation matrix $R$ via `scipy.linalg.orthogonal_procrustes` aligns image and text spaces.
   - Procrustes rotation narrows the modality gap and preserves/improves downstream classification performance.
