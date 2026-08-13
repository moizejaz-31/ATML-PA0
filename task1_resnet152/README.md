# Task 1 — Inner Workings of ResNet-152

## Objectives & Key Findings

1. **Baseline Fine-Tuning:**
   - Pretrained ImageNet representations transfer efficiently to CIFAR-10 with head-only fine-tuning.
   - Training a 60M parameter network from scratch on small datasets leads to severe overfitting and computational overhead.

2. **Residual Connection Ablation:**
   - Disabling skip connections in ResNet Bottleneck blocks degrades gradient flow.
   - Without identity shortcuts, training convergence slows drastically and validation accuracy declines.

3. **Feature Hierarchies:**
   - Forward hooks capture activation patterns from `layer1` through `avgpool`.
   - Early layers encode generic low-level features (edges, textures), while deep layers show clear class separability in t-SNE/UMAP projections.

4. **Transfer Learning Comparison:**
   - Pretrained initialization outperforms random initialization by a significant margin.
   - Unfreezing `layer4` alongside the head provides optimal accuracy-to-compute efficiency.
