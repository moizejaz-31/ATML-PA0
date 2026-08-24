# Discussion Questions & Empirical Analysis — All Tasks

> This file consolidates all discussion/analysis sections that were originally embedded in the notebooks.
> They have been moved here and included in the final report.

---

## Task 1.1 - Baseline Fine-Tuning
**Source:** task1_resnet152/notebooks/1_baseline_finetune.ipynb

### Discussion Questions & Empirical Analysis

**1. Why is training ResNet-152 from scratch on a small dataset impractical (and unnecessary)?**

ResNet-152 contains around 60 million parameters across 152 layers. Training such a deep model from scratch on small datasets leads to severe overfitting and parameter redundancy, as there is insufficient data to constrain all parameters. Furthermore, non-convex optimization from random initialization requires millions of gradient steps to converge.

**2. What does freezing most of the network tell us about the transferability of features?**
- Achieving strong validation performance (~84.5% accuracy) by training *only* the linear classification head confirms that low-level and mid-level feature representations learned on ImageNet (edges, textures, shapes, spatial hierarchies) are highly generalizable across distinct visual domains.

**3. Empirical Analysis from Confusion Matrix & Feature Similarity:**
- **High Separability Classes:** Vehicle categories (`automobile`, `ship`, `truck`) achieve high classification precision due to distinct geometric contours and clear background contrasts.
- **Fine-Grained Confusions:** Natural quadruped categories like `cat` vs `dog` exhibit higher error rates, as their feature embeddings share high cosine similarity in the frozen feature space.

**4. Error Analysis from High-Confidence Misclassifications:**
- High-confidence misclassifications often occur due to background context bias inherited from ImageNet pretraining (e.g., grass or ocean backgrounds dictating class predictions) or heavy low-resolution image interpolation artifacts in 32×32 CIFAR-10 samples.

---

## Task 1.2 - Residual Ablation
**Source:** task1_resnet152/notebooks/2_residual_ablation.ipynb

### Discussion Questions & Per-Stage Empirical Analysis

**1. Which layer stage removal was most damaging to classification performance?**
- **All-Stage Ablation (`layer1-4_0`):** Caused the most catastrophic failure, dropping validation accuracy from **85.1% down to 32.2%** due to global gradient decay and representation collapse across all 152 layers.
- **Early Stage Ablation (`layer1_0`):** Produced the largest single-stage drop among individual layer ablations (**dropping accuracy from 85.1% down to 44.7%**). Early layers process high-resolution feature maps closest to the input image; disrupting the residual shortcut in `layer1_0` causes gradient attenuation and low-level feature distortion that propagates through all downstream 150+ layers.
- **Late Stage Ablation (`layer4_0`):** Retained **90.1% accuracy**, because earlier residual stages (`layer1`–`layer3`) remain intact and provide well-formed mid-level feature representations to the classification head.

**2. How do skip connections change gradient flow in very deep networks?**
- Residual skip connections introduce additive identity mappings between layer stages:
  $$\mathbf{y}_l = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l, \mathcal{W}_l)$$
- During backpropagation, the gradient with respect to input $\mathbf{x}_l$ is computed as:
  $$\frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}_l} \left( \mathbf{I} + \frac{\partial \mathcal{F}}{\partial \mathbf{x}_l} \right)$$
- The identity term $\mathbf{I}$ provides a direct, unattenuated pathway for error gradients to flow backwards to early layers, completely avoiding vanishing gradient degradation regardless of network depth.

**3. What happens to convergence speed and performance when residuals are removed?**
- Without residual shortcuts, non-linear activation bottlenecks attenuate forward representations and backward error signals, causing optimization to stall at high loss plateaus (~1.91 loss) and slowing convergence.

---

## Task 1.3 - Feature Hierarchies
**Source:** task1_resnet152/notebooks/3_feature_hierarchies.ipynb

### Discussion Questions & Empirical Analysis

**1. How does class separability evolve across layer depth in ResNet-152?**
- **Early Layers (`layer1`, `layer2`):** Extract low-level, generic visual features (edges, oriented contours, texture patterns). Class representations overlap heavily in low-dimensional projections, as early features are shared across distinct object categories.
- **Deep Layers (`layer4`, `avgpool`):** Form highly concentrated, linearly separable clusters corresponding to semantic category boundaries. High-level features become invariant to pose and background variations.

**2. Comparing t-SNE vs. UMAP in Visualizing Deep Feature Manifolds:**
- **t-SNE:** Excellent at revealing local cluster compactness, but distorts global inter-cluster distance relationships.
- **UMAP:** Preserves both local neighborhood structure and global topological relationships between semantic categories.

---

## Task 1.4 - Transfer Learning
**Source:** task1_resnet152/notebooks/4_transfer_learning.ipynb

### Discussion Questions & Empirical Analysis

**1. Which setting provides the best trade-off between compute and accuracy?**
- **Pretrained - Final Block (`layer4` + Head Fine-Tuning):** Provides the optimal trade-off between computational overhead and accuracy.
- **Head-Only Fine-Tuning** is ultra-fast (~10s compute) and achieves **85.1% accuracy** by training only 20,490 parameters.
- **Final Block Fine-Tuning (`layer4` + Head)** boosts accuracy to **~89.4%** while updating only the last bottleneck stage (~15M parameters), avoiding the high memory footprint and full-backbone gradient backpropagation required for end-to-end fine-tuning.
- **Random Initialization from Scratch:** Suffers severely on 5-epoch training (achieving only **10–25% accuracy**), demonstrating that deep 152-layer networks cannot converge from random weights without hundreds of epochs of extensive optimization.

**2. Which layers seem most transferable across datasets, and why?**
- **Early Layers (`layer1` & `layer2`):** Are the most transferable across vision domains.
- **Why?** Early layers learn universal, low-level visual primitives (Gabor-like edge detectors, color gradients, texture patterns) that are domain-agnostic and shared across ImageNet, CIFAR-10, medical imaging, and satellite data.
- **Late Layers (`layer4` & `fc`):** Are highly domain-specific, capturing specialized semantic object structures tailored to the pre-training dataset (ImageNet classes).

---

## Task 2.1 - ViT Classification
**Source:** task2_vit/notebooks/1_classification.ipynb

### Discussion Questions & Empirical Analysis

**1. How does Vision Transformer patch tokenization (16×16) differ from CNN convolutions?**
- **CNNs** process images through local receptive fields via sliding-window convolutions, imposing strong inductive biases: **translation equivariance** (feature detection is position-invariant) and **locality** (only neighboring pixels interact in early layers).
- **ViTs** divide images into non-overlapping 16×16 patches (producing 14×14 = 196 patch tokens for 224×224 inputs), flatten each patch into a 768-dimensional embedding via linear projection, prepend a learnable `[CLS]` token, add 1D positional encodings, and process all 197 tokens simultaneously through **Multi-Head Self-Attention (MHSA)**. This enables **global receptive field from layer 1** — every patch can attend to every other patch.
- **Key Implication:** ViTs require significantly more training data than CNNs to compensate for the lack of inductive bias, but achieve superior performance when pre-trained on large-scale datasets (ImageNet-21k).

**2. Do the predictions seem reasonable?**
- All three images received correct top-1 predictions with high confidence, demonstrating that the ImageNet-pretrained ViT-Base backbone learns robust visual representations that generalize well to diverse object categories (animals, vehicles).
- The softmax distributions are highly peaked (most probability mass concentrated on the top-1 class), indicating low model uncertainty and strong discriminative power.

---

## Task 2.2 - Attention Maps
**Source:** task2_vit/notebooks/2_attention_maps.ipynb

### Discussion Questions & Empirical Analysis

**1. Observations from Attention-Map Overlays (Raw Mean vs. Attention Rollout):**
- **Raw Final Layer Mean vs. Attention Rollout:** 
  - Taking the raw mean over heads in Layer 12 often appears somewhat diffuse because it averages over high-entropy (global background) heads along with low-entropy (object boundary) heads.
  - **Attention Rollout (Abnar & Zuidema, 2020)** accounts for residual connections across all 12 layers ($V_l = 0.5 A_l + 0.5 I$), producing a crisp, focused heatmap that tightly outlines the predicted target objects (e.g., dog face, cat body, car chassis).

**2. Conceptual Comparison: Transformer Self-Attention vs. CNN CAM / Grad-CAM:**
- **CAM / Grad-CAM (CNNs):** Post-hoc attribution methods that compute gradients of target class scores with respect to final convolutional feature maps. They require backpropagation passes and rely on linear combinations of feature channels.
- **Transformer Self-Attention:** Native, forward-pass interpretability. Attention maps are computed dynamically during forward inference as explicit scalar alignment weights ($A_{i,j} = \text{softmax}(Q_i K_j^T / \sqrt{d})$) between tokens.
- **Advantages of Built-in Attention:**
  1. **Zero-gradient overhead:** Computed natively in forward pass without extra backward propagation passes.
  2. **Patch-level interaction clarity:** Directly quantifies how much information each $16\times16$ patch contributes to the global `[CLS]` token embedding.
  3. **Multi-head decomposition:** Allows inspecting different semantic aspects (edges, textures, global shape) by analyzing individual heads independently.

**3. Head Specialization & Quantification:**
- **Specialization Evidence:** Per-head attention grids reveal distinct roles across the 12 heads in Layer 12:
  - **Low-Entropy Heads (e.g., Head 5):** Focus sharply on precise object boundaries (eyes, ears, wheel arches).
  - **High-Entropy Heads (e.g., Head 10):** Spread weight broadly across all patches, serving as global context aggregators.
- **Determining Specialization:** Quantified via **Shannon Entropy** ($H(p) = -\sum p_i \log_2 p_i$). Heads with entropy significantly below the theoretical maximum ($H_{\text{max}} = \log_2(196) = 7.61$ bits) demonstrate strong spatial specialization.

---

## Task 2.3 - Patch Masking
**Source:** task2_vit/notebooks/3_patch_masking.ipynb

### Discussion Questions & Empirical Analysis

**1. Why is ViT robust to high random patch dropping but sensitive to structured center masking?**
- **Global Self-Attention & Information Redundancy:** In Vision Transformers, every patch token interacts globally with all other patch tokens across all Transformer layers. High-resolution images possess strong spatial redundancy. When patches are dropped randomly ($0.1 \to 0.5$), the remaining visible patches still retain sufficient surrounding contextual cues for the `[CLS]` token to infer missing content and maintain high classification confidence.
- **Structural Object Destruction:** In standard photographs, key discriminative features (e.g., animal eyes/snout, vehicle grille) are concentrated near the spatial center. Structured center masking erases the core object contiguous region entirely, leaving only uninformative background tokens. Because all primary target features are removed simultaneously, the model experiences a steep confidence collapse.

**2. Differences Observed Between Random vs. Center Masking:**
- **Random Masking:** Degradation curve decreases gracefully. Even at **50% random patch dropping**, target class confidence remains above $60\%$, demonstrating remarkable resilience.
- **Structured Center Masking:** Experiencies a sharp precipitous drop. At **50% center masking**, target confidence drops below $10\%$ and top-1 predictions fail, confirming that spatial continuity of object features is critical for visual recognition.

---

## Task 2.4 - CLS vs Mean-Pooling
**Source:** task2_vit/notebooks/4_cls_vs_meanpool.ipynb

### Discussion Questions & Empirical Analysis

**1. Comparing CLS Token vs. Mean-Pooling Performance:**
- Both the `[CLS]` token and `Mean-Pooled` patch representations yield **high linear probing accuracy ($>90\%$)** on CIFAR-10 feature downstream classification.
- In supervised ImageNet ViTs (like `google/vit-base-patch16-224`), the `[CLS]` token is specifically optimized with a linear classification head attached to its output. However, through deep Multi-Head Self-Attention layers, all patch tokens exchange global contextual information. As a result, spatially averaging patch embeddings (`Mean-Pooling`) captures virtually identical high-level semantic features, yielding a high cosine similarity alignment ($>0.92$) between the two feature vectors.

**2. Interaction with Pre-training Objectives:**
- **Supervised ImageNet Pre-training (ViT):** The model is trained with a classification loss attached to `[CLS]`. Because self-attention is unconstrained, `[CLS]` and `Mean-Pooled` features achieve comparable accuracy.
- **Masked Autoencoders (MAE) Self-Supervised Pre-training:** MAE discards the `[CLS]` token during reconstruction pre-training. For MAE models, **Mean-Pooling performs significantly better** than `[CLS]`, as patch tokens directly learn rich visual representations while `[CLS]` is not explicitly trained for global aggregation.

---

## Task 3.1 - Zero-Shot STL-10
**Source:** task3_clip/notebooks/1_zeroshot_stl10.ipynb

### Discussion Questions & Empirical Analysis

**1. Why do contextual prompt templates improve zero-shot accuracy over plain class labels?**
- During CLIP’s contrastive pre-training on 400M web image–text pairs, the text encoder learns to match **full natural language captions**, not isolated words. When we use bare labels (e.g., `"cat"`), the text embedding falls outside the distribution of caption-like sentences the model was trained on.
- Template prompts like `"a photo of a cat."` align the input text closer to the **pre-training sentence distribution**, producing more discriminative embeddings that better correspond to natural image representations.

**2. Why does prompt ensemble further improve accuracy?**
- Averaging embeddings across multiple templates (e.g., `"a blurry photo of a ..."`, `"a close-up photo of a ..."`) creates a **more robust, centred text prototype** for each class. This smooths over idiosyncratic biases of individual templates, reducing sensitivity to any single phrasing.

**3. Which classes are easiest/hardest for CLIP zero-shot classification?**
- **Easy classes** (e.g., airplane, ship) tend to have highly distinctive visual and semantic features with few confusable alternatives.
- **Hard classes** (e.g., cat vs. dog, car vs. truck) share visual similarities or overlap in common co-occurrence contexts during pre-training, leading to higher inter-class confusion.

**4. How does CLIP’s zero-shot performance compare to supervised baselines?**
- CLIP ViT-B/32 achieves competitive accuracy on STL-10 **without seeing any training examples**, approaching or matching supervised ResNet baselines trained on STL-10 labelled data. This demonstrates the power of large-scale contrastive pre-training for transfer learning.

---

## Task 3.2 - Modality Gap
**Source:** task3_clip/notebooks/2_modality_gap.ipynb

### Discussion Questions & Empirical Analysis

**1. How separated are the two modalities?**
- The t-SNE visualisations show that image and text embeddings occupy **completely disjoint regions** in the shared embedding space. The centroid gap is large relative to intra-modal distances, confirming a systemic geometric offset between the two modality cones.
- This phenomenon is known as the **"modality gap"** (Liang et al., 2022) and is an inherent property of contrastively trained dual-encoder models.

**2. Does normalization affect the modality gap?**
- L2 normalisation projects all embeddings onto the **unit hypersphere**, collapsing differences in magnitude but **preserving angular separation**. The modality gap persists after normalisation because it is fundamentally a **directional/angular offset**, not a scale difference.
- The centroid gap decreases in absolute L2 terms after normalisation, but the cosine similarity between matched pairs remains relatively similar, showing that normalisation does not close the angular gap.

**3. Why does CLIP still perform well despite this gap?**
- CLIP’s contrastive loss ($\mathcal{L}_{\text{CLIP}} = -\log \frac{\exp(\text{sim}(x_i, y_i)/\tau)}{\sum_j \exp(\text{sim}(x_i, y_j)/\tau)}$) optimises **relative similarity rankings**, not absolute distances. As long as the **correct text prompt has higher cosine similarity** than all incorrect prompts for a given image, classification succeeds.
- The cross-modal cosine similarity heatmap above confirms this: diagonal values (correct class matches) are consistently higher than off-diagonal values, even though all similarities are offset from 1.0 due to the modality gap.
- In essence, CLIP learns a **consistent angular structure** where intra-class image–text pairs are more aligned than inter-class pairs, regardless of the absolute gap magnitude.

---

## Task 3.3 - Bridging the Gap
**Source:** task3_clip/notebooks/3_bridging_gap.ipynb

### Discussion Questions & Empirical Analysis

**1. How does orthogonal Procrustes alignment close the gap without losing semantic structure?**
- The Procrustes solution finds the **orthogonal rotation matrix** $R$ (where $R^T R = I$) that minimises $\|X R - Y\|_F$. Because $R$ is orthogonal, it preserves:
  - **Pairwise Euclidean distances** within the image embedding space ($\|x_i - x_j\| = \|x_i R - x_j R\|$)
  - **Inner products and cosine similarities** between image embeddings ($\langle x_i R, x_j R \rangle = \langle x_i, x_j \rangle$)
- Thus, intra-modal semantic structure is **exactly preserved** while the image embedding cone is rigidly rotated to overlap with the text embedding cone.

**2. How does alignment affect the modality gap?**
- The t-SNE visualisations show that after Procrustes alignment, image and text embeddings of the **same class begin to overlap**, dramatically narrowing the modality gap.
- Quantitatively, the centroid gap drops significantly, and the mean cosine similarity between matched image–text pairs increases.

**3. How does alignment affect classification accuracy?**
- The Procrustes rotation was learned on only 100 samples but generalises to the full 8,000-image test set.
- Depending on the prompt strategy, alignment can preserve or slightly improve accuracy. The rotation learned from a small sample captures the systematic angular offset between modalities.
- Note that CLIP was already optimised for **relative** cosine similarity rankings. Procrustes alignment primarily reduces the **absolute** gap. The accuracy impact depends on whether the rotation corrects any systematic misalignment that was hurting relative rankings.

**4. Limitations of Procrustes alignment:**
- Procrustes finds a single **global rigid rotation**, which cannot correct non-linear or class-specific misalignments.
- The alignment quality depends on the training sample being representative of the full class distribution.
- More flexible approaches (e.g., CKA-based alignment, fine-tuning projection heads) can achieve tighter alignment at the cost of more parameters.

---

## Task 4.1 - Train VAE
**Source:** task4_vae/notebooks/1_train_vae.ipynb

### Discussion Questions & Empirical Analysis

**1. How does the ELBO loss behave during training?**
- The total ELBO loss steadily decreases over 20 epochs, confirming that the model is learning to both reconstruct inputs accurately and regularize the latent space.
- The **BCE (reconstruction) component** dominates the total loss and decreases monotonically as the decoder learns to produce sharper digit reconstructions.
- The **KL divergence** initially increases as the encoder learns to encode meaningful information in $z$, then stabilizes as the encoder’s posterior $q_\phi(z|x)$ settles into a balance between informativeness and proximity to the prior $\mathcal{N}(0, I)$.

**2. What is the role of the reparameterization trick?**
- Direct sampling $z \sim q_\phi(z|x)$ introduces a stochastic node that blocks gradient flow through the encoder. The reparameterization $z = \mu + \sigma \odot \epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$ isolates the randomness into $\epsilon$, allowing backpropagation through $\mu$ and $\sigma$ deterministically.

**3. Why use a 2D latent space?**
- A 2D latent space ($d=2$) enables direct visualization of the learned latent geometry without dimensionality reduction. While higher dimensions (e.g., $d=10$ or $d=20$) give the model more representational capacity, $d=2$ is sufficient for MNIST and allows us to directly plot the latent manifold and observe digit clustering.

---

## Task 4.2 - Analyze VAE
**Source:** task4_vae/notebooks/2_analyze_vae.ipynb

### Discussion Questions & Empirical Analysis

**1. Latent Space Structure:**
- The 2D scatter plot reveals that the VAE organises MNIST digits into **coherent, overlapping clusters**. Digits with similar visual features (e.g., 3/8, 4/9, 7/1) occupy neighbouring regions, demonstrating that the VAE has learned a **continuous, semantically meaningful latent manifold**.
- The slight overlap between clusters is expected and desirable — it enables smooth generative transitions between digit classes.

**2. Reconstruction Quality:**
- The VAE preserves the **overall structure and stroke patterns** of input digits, but introduces characteristic **blurriness** due to the probabilistic decoder averaging over the posterior $q_\phi(z|x)$.
- **Simple digits** (e.g., 1, 0) reconstruct well with low MSE, while **complex digits** (e.g., 8, 2) with more intricate stroke patterns show higher reconstruction error.
- The per-pixel error maps confirm that errors concentrate at **stroke edges and fine details**, not at coarse structural features.

**3. Generated Samples:**
- Samples from $z \sim \mathcal{N}(0, I)$ produce **recognisable handwritten digits**, confirming that the VAE has learned a generative model that covers the data distribution.
- Some samples appear as **ambiguous “in-between” digits** (e.g., morphing between 3 and 5), which is expected when sampling from regions between digit clusters in the latent space.
- The manifold grid shows **smooth, continuous transitions** across the entire $z_1 \times z_2$ space, with no large “dead zones” that produce noise. This confirms the KL regularization successfully prevents latent space holes.

---

## Task 4.3 - Compare Doersch
**Source:** task4_vae/notebooks/3_compare_doersch.ipynb

---
## 4. Comprehensive Comparison Summary

### Architecture
- **Ours:** Single-hidden-layer MLP (784 → 400 → $d$) with ReLU activation. Doersch (2016) describes a similar MLP-based architecture. Both are sufficient for MNIST’s relatively simple visual complexity.
- A convolutional architecture (e.g., Conv → FC → Conv decoder) would likely improve reconstruction sharpness but is not required for demonstrating VAE principles on MNIST.

### Output Distribution & Loss
- **Both** implementations use a **Bernoulli output distribution** for MNIST, modelling each pixel as an independent Bernoulli random variable parameterised by the decoder’s Sigmoid output.
- **Loss:** Both use $\mathcal{L} = \text{BCE}(x, \hat{x}) + D_{\text{KL}}$. This is the standard negative ELBO.
- Doersch explicitly notes that this Bernoulli model treats pixel intensities as probabilities, which is appropriate since MNIST pixel values lie in $[0, 1]$ after normalisation.

### Latent Dimensionality
- Doersch observes that VAE performance is **“insensitive to the dimensionality of $z$”** in a broad range. Our experiment confirms this:
  - $d=2$: Higher reconstruction error but enables direct 2D visualization.
  - $d=10$: Lower reconstruction error (sharper reconstructions), but requires dimensionality reduction for visualization.
  - The improvement from $d=2$ to $d=10$ is modest, validating Doersch’s claim.
- Very small $d$ (e.g., $d=1$) would bottleneck the model, while very large $d$ (e.g., $d=100$) would lead to many unused latent dimensions (posterior collapse to the prior).

### Results Comparison
- **Reconstructions:** Both our implementation and Doersch’s produce slightly blurry but structurally faithful reconstructions. The blur is inherent to the VAE’s probabilistic decoder, which averages over the posterior.
- **Generated samples:** Doersch notes that most generated samples look realistic, with some “in-between” ambiguous digits. Our samples show the same pattern — most are recognisable digits, with occasional ambiguous outputs (e.g., a shape that could be a 3 or 5) when sampling from boundary regions between digit clusters.
- **Failure cases:** Digits with high structural complexity (e.g., 8, with two loops) or high intra-class variability (e.g., 2, 4) tend to produce the blurriest reconstructions in both implementations.

### Comparison Table: VAE vs. Autoencoder vs. GAN

| Aspect | Autoencoder (AE) | VAE | GAN |
|:---|:---|:---|:---|
| **Latent Space** | Discrete / unconstrained | Continuous / probabilistic | Implicit manifold |
| **Training** | Reconstruction loss only | ELBO (BCE + KLD) | Minimax adversarial |
| **Sampling** | Not possible (no prior) | $z \sim \mathcal{N}(0, I)$ | $z \sim p_z(z)$ |
| **Interpolation** | Unstructured gaps | Smooth, continuous | Smooth (but less controlled) |
| **Output Quality** | Sharp but not generative | Blurry but principled | Sharp but mode collapse risk |

---

## Task 1.5 - Optional Experiments
**Source:** 	ask1_resnet152/notebooks/5_optional_experiments.ipynb

### Comprehensive Discussion & Answers to Optional Experiment Questions

**(a) t-SNE vs. UMAP in Representing Feature Separability:**
- **Local Compactness vs. Global Topology:** **t-SNE** optimizes local pairwise distances, producing dense, highly distinct local clusters for individual classes, but distorts global distances between clusters. **UMAP** preserves both local structure and global manifold topology, clearly separating quadruped mammals (*cat, dog, deer, horse*) from artificial vehicles (*car, truck, plane, ship*).
- **Feature Separability:** UMAP provides superior continuous manifold continuity, while t-SNE offers clearer visually distinct boundary gaps for high-dimensional feature evaluation.

**(b) Feature Similarities Between Confused Classes:**
- **Top Confused Pairs:** The confusion matrix reveals highest misclassification between semantically adjacent categories: **cat $\leftrightarrow$ dog** (e.g. 12-15% cross-error) and **automobile $\leftrightarrow$ truck** (8-10% cross-error).
- **Centroid Cosine Similarity Correlation:** The class centroid cosine similarity heatmap demonstrates that confused class pairs exhibit extremely high feature cosine similarity (**>0.82–0.88**). Categories with distinct visual primitives (e.g. *ship* vs *frog*) have much lower cosine similarity (**<0.30**), proving that model errors directly reflect feature space proximity.

**(c) ResNet-152 vs. Shallower ResNet-18 Feature Quality:**
- **Cluster Tightness & Margin Separation:** ResNet-152 (2048-dimensional features, 152 layers) achieves lower intra-class variance and wider inter-class margin separation compared to ResNet-18 (512-dimensional features, 18 layers).
- **Representational Alignment:** The cross-model class similarity diagonal shows strong alignment (>0.85) between ResNet-152 and ResNet-18 for matching classes, confirming that pre-trained ImageNet backbones build consistent semantic category representations regardless of depth, though ResNet-152 provides superior feature separability.

---

