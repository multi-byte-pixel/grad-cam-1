# Grad-CAM Terminology and Data Flow

## Core Concepts Table

| Term | Definition | Role in Grad-CAM | Shape Example (ResNet-50) | Key Properties |
|------|------------|------------------|---------------------------|----------------|
| **Logits / Class Score** | Raw output from the final linear layer before softmax; often denoted `y^c` for class `c` | The target quantity we differentiate. High score = model's confidence for class `c` before normalization | `(batch, 1000)` | Pre-softmax; unbounded; monotonically related to probability |
| **Softmax Probability** | `softmax(y^c) = exp(y^c) / Σ exp(y^i)` | Not directly used in Grad-CAM (we differentiate logits, not probabilities) | `(batch, 1000)` | Sums to 1 across classes; non-linear |
| **Convolutional Feature Maps** | Spatial activation tensor from a conv layer; denoted `A^k` for channel `k` | Source of spatial localization information. High values = strong feature presence at spatial location `(i,j)` | `(batch, 512, 7, 7)` for layer4 | 3D tensor: channels × height × width |
| **Gradients w.r.t. Feature Maps** | `∂y^c / ∂A^k_{ij}` — how much a small change in activation `(k, i, j)` changes class score | Indicates importance: high gradient = this activation strongly influences the target class | Same as feature maps | Can be positive or negative |
| **Channel Weights (α^k)** | Aggregated importance score for channel `k`. In Grad-CAM: global-average-pool of gradients | Determines how much channel `k` contributes to the final heatmap | `(512,)` for 512 channels | One scalar per channel |
| **Raw CAM** | Weighted sum of feature maps: `L = Σ_k α^k · A^k` | Produces a spatial map showing class-relevant regions before ReLU | `(batch, 7, 7)` | Can have negative values |
| **ReLU CAM** | `ReLU(Raw CAM)` — zero out negative values | Keeps only "positive evidence" (features that increase class score). Discards suppressors | `(batch, 7, 7)` | Non-negative; interpretable as "supporting evidence" |
| **Upsampling** | Resize heatmap from layer resolution (e.g., 7×7) to input resolution (224×224) | Makes heatmap align with input pixels for visualization | `(batch, 224, 224)` | Bilinear interpolation typical; can create artifacts |
| **Overlay** | Blend upsampled heatmap with original image (e.g., using colormap and alpha blending) | Visual representation for human interpretation | RGB image `(224, 224, 3)` | Purely for visualization; not used in computation |

---

## Forward/Backward Data Flow Pipeline

### Forward Pass (Inference)
```
Input Image (224×224×3)
   ↓ [normalize with ImageNet stats]
Normalized Tensor (1×3×224×224)
   ↓ [conv layers: layer1 → layer2 → layer3 → layer4]
Feature Maps A (1×512×7×7)  ← **HOOK: capture activations**
   ↓ [adaptive avg pool + flatten]
Flattened Features (1×2048)
   ↓ [fully connected layer]
Logits y^c (1×1000)  ← **TARGET: class c score**
   ↓ [optional: softmax for display only]
Class Probabilities (1×1000)
```

### Backward Pass (Gradient Computation)
```
Target Class Score y^c (scalar)
   ↓ [compute ∂y^c / ∂A]
Gradients ∂y^c/∂A (1×512×7×7)  ← **HOOK: capture gradients**
   ↓ [Grad-CAM: global average pool across spatial dims]
Channel Weights α^k (512,)
   ↓ [Grad-CAM++: pixel-wise alpha weighting before pooling]
Alternative Weights w^k (512,)
```

### Heatmap Generation
```
Activations A (1×512×7×7)
Channel Weights α (512,)
   ↓ [weighted sum: Σ α^k · A^k]
Raw CAM (1×7×7)
   ↓ [ReLU: zero negative values]
ReLU CAM (1×7×7)
   ↓ [normalize to [0, 1]]
Normalized Heatmap (1×7×7)
   ↓ [bilinear upsample to input size]
Full-Res Heatmap (1×224×224)
   ↓ [apply colormap + alpha blend with input]
Overlay Visualization (224×224×3 RGB)
```

---

## Why the Last Spatial Convolutional Layer?

### Trade-off: Semantic Meaning vs Spatial Resolution

| Layer Depth | Spatial Resolution | Semantic Level | Grad-CAM Behavior |
|-------------|-------------------|----------------|-------------------|
| **Early (layer1)** | High (56×56) | Low (edges, textures) | Heatmap is detailed but not class-specific; highlights generic features |
| **Middle (layer2, layer3)** | Medium (28×28, 14×14) | Medium (parts, patterns) | Better class discrimination but still somewhat generic |
| **Late (layer4)** | Low (7×7) | High (object-level) | Strong class specificity, but coarse localization |
| **FC layers** | None (0D) | Highest (decision) | No spatial structure — cannot generate heatmap |

**Typical choice:** The last convolutional block before global pooling (e.g., ResNet's `layer4[-1]`, VGG's final conv layer).

**Rationale:**
1. **Semantic meaning:** Late layers encode object-level concepts strongly correlated with class decisions
2. **Spatial structure:** Still has spatial dimensions (unlike FC layers), enabling localization
3. **Empirical performance:** Validated in Grad-CAM/Grad-CAM++ papers

---

## Architecture-Specific Target Layers

Different architectures organize spatial feature extraction differently. Here's where to hook:

| Architecture | Typical Target Layer | Why | Output Shape |
|--------------|---------------------|-----|--------------|
| **ResNet-50** | `model.layer4[-1]` | Last bottleneck block before adaptive pool | `(B, 2048, 7, 7)` |
| **VGG-16** | `model.features[30]` (final conv) | Last conv before flatten | `(B, 512, 7, 7)` |
| **MobileNetV2** | `model.features[-1]` | Last inverted residual block | `(B, 1280, 7, 7)` |
| **EfficientNet** | `model.features[-1]` | Top conv block | Varies by model scale |
| **Vision Transformer (ViT)** | Reshape attention output | No true conv layers; requires custom transform | Patch tokens reshaped |

**Note:** For transformers, `pytorch-grad-cam` provides `reshape_transform` utilities to map attention outputs back to 2D spatial structure.

---

## Common Misconceptions and Clarifications

### ❌ "High heatmap intensity = that pixel caused the prediction"
**Reality:** Heatmaps show correlation, not causation. High gradient + high activation means that feature is important to the *model's decision*, not that it represents true causal factors.

### ❌ "Grad-CAM works on the fully connected layer"
**Reality:** FC layers have no spatial structure. Grad-CAM requires convolutional (spatial) layers.

### ❌ "Sharper heatmaps are always better"
**Reality:** Resolution is constrained by the target layer. Upsampling interpolates; it doesn't add information. A tight heatmap on a misclassified image is still wrong.

### ❌ "ReLU removes useless information"
**Reality:** ReLU removes *negative evidence* (features that suppress the class). This information can be valuable for debugging, but Grad-CAM focuses on "what supports the prediction."

### ✅ "Grad-CAM is a model diagnostic, not ground truth"
**Correct:** Use it to inspect model behavior, not to make claims about the real world.

---

## Next: Checkpoint Questions for You

Before proceeding to Block 2, consider:

1. **What quantity are we differentiating, with respect to what tensor?**
   - (Hint: target class score y^c, with respect to feature map activations A^k)

2. **Why is a late convolutional layer a trade-off between semantic meaning and spatial resolution?**
   - (Hint: deeper = more semantic but coarser; shallower = finer resolution but less class-specific)

3. **What will count as a successful explanation, versus merely a visually attractive overlay?**
   - (Hint: reproducibility, consistency with known model behavior, falsifiability — not just aesthetic appeal)

---

## References
- Grad-CAM paper Section 3 (Method)
- `pytorch_grad_cam/base_cam.py` (implementation of hooks and aggregation)
- `pytorch_grad_cam/grad_cam.py` (global average pooling logic)
