# Paper Map: Grad-CAM and Grad-CAM++

## Paper 1: Grad-CAM (ICCV 2017)

**Full citation:** Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." *2017 IEEE International Conference on Computer Vision (ICCV)*, pp. 618-626.

### Research Question
How can we generate visual explanations for decisions from CNN-based models (classification, captioning, VQA) without architectural changes or retraining?

### Method
1. **Input:** A trained CNN, an input image, and a target class (or neuron)
2. **Forward pass:** Capture activations `A^k` from a target convolutional layer
3. **Backward pass:** Compute gradients of the target class score with respect to those activations
4. **Weight computation:** Global-average-pool the gradients to get per-channel importance weights `α^k`
5. **Weighted combination:** Sum weighted activations across channels: `L = Σ α^k · A^k`
6. **Positive evidence:** Apply ReLU to keep only positive contributions
7. **Visualization:** Upsample to input resolution and overlay as a heatmap

**Key equation (Eq. 1):**
```
α^k_c = (1/Z) Σ_i Σ_j  ∂y^c / ∂A^k_ij
```
where `y^c` is the class score (pre-softmax), `A^k` is the k-th feature map, and Z is the number of pixels.

### Evidence
- Evaluated on ImageNet classification (VGG-16, ResNet)
- Applied to image captioning and VQA tasks
- Human studies: Grad-CAM localizations align better with annotator trust than gradient-only methods
- Ablation: Grad-CAM alone is coarse; combining with Guided Backprop (Guided Grad-CAM) improves fine-grained detail

### Limitations
1. **Coarse resolution:** Heatmap resolution is limited by the target layer's spatial dimensions (e.g., 7×7 for ResNet layer4)
2. **Single-object bias:** Global-average pooling may not handle multiple instances of the same class well
3. **Positive-only:** ReLU discards negative evidence (features that suppress a class)
4. **Not a proof of causality:** High activation + high gradient ≠ causal relationship
5. **Layer dependence:** Results vary significantly with target layer choice

---

## Paper 2: Grad-CAM++ (WACV 2018, IEEE 8354201)

**Full citation:** Aditya Chattopadhyay, Anirban Sarkar, Prantik Howlader, and Vineeth N. Balasubramanian. "Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks." *2018 IEEE Winter Conference on Applications of Computer Vision (WACV)*, pp. 839-847. DOI: 10.1109/WACV.2018.00097

### Research Question
Can we improve Grad-CAM's localization quality, especially for multiple occurrences of the same class in one image?

### Method
Grad-CAM++ refines the weight computation to use **pixel-wise weighting** instead of global averaging:

1. Compute **alpha coefficients** `α^k_ij` for each spatial location `(i,j)` in channel `k`
2. Weight gradients by these alpha values before aggregating
3. Alpha depends on positive partial derivatives and second-order gradient information

**Key equations (simplified from paper):**
```
α^k_ij = (∂²y^c / ∂(A^k_ij)²) / [2·(∂²y^c / ∂(A^k_ij)²) + Σ_a,b A^k_ab · (∂³y^c / ∂(A^k_ij)³)]

w^k = Σ_i,j α^k_ij · ReLU(∂y^c / ∂A^k_ij)

L^c_Grad-CAM++ = ReLU( Σ_k w^k · A^k )
```

The alpha term captures **how much each pixel contributes to the gradient**, accounting for curvature (second derivative).

### Evidence
- Evaluated on ImageNet (VGG-16, ResNet-50)
- Multi-instance images: Grad-CAM++ highlights all instances; Grad-CAM highlights only one or averages
- Object localization: Higher overlap with bounding boxes
- Failure cases: Also benefits from combining with fine-grained methods (Guided Grad-CAM++)

### Claimed Improvements Over Grad-CAM
1. **Better multi-instance localization**
2. **Sharper/more concentrated heatmaps** (less diffuse)
3. **Improved localization metrics** (e.g., Average Drop, Increase in Confidence)

### Limitations
1. **Increased computation:** Requires second and third derivatives (approximated in practice)
2. **Still coarse resolution:** Same spatial bottleneck as Grad-CAM
3. **ReLU still applied:** Discards negative evidence
4. **Localization ≠ explanation faithfulness:** A tighter heatmap doesn't guarantee the model actually uses those pixels causally
5. **Overinterpretation risk:** Visually appealing overlays can mislead stakeholders into overclaiming model reasoning

---

## Comparison Table

| Aspect | Grad-CAM | Grad-CAM++ |
|--------|----------|------------|
| **Weight computation** | Global-average pooling of gradients | Pixel-wise alpha-weighted gradients |
| **Multi-instance handling** | Averages across all instances | Claims to localize each instance |
| **Computational cost** | Lower (first-order gradients) | Higher (second/third-order derivatives) |
| **Localization sharpness** | More diffuse | More concentrated |
| **Equation complexity** | Simple (one sum) | More complex (alpha coefficients) |
| **Primary use case** | Single dominant object | Multiple instances of same class |

---

## What Both Methods Do NOT Claim
- **Causal explanations:** Heatmaps show correlation, not causation
- **Pixel-level precision:** Both are constrained by target layer resolution
- **Robustness:** Small input perturbations can change heatmaps significantly
- **Architecture-agnostic perfection:** Performance varies by model, layer, task

---

## Open Questions for This Project
1. On our test images, is Grad-CAM++ visibly better than Grad-CAM?
2. How do heatmaps change when targeting a non-top-1 class?
3. What happens when the target layer is too early (high resolution, low semantics) or too late (no spatial structure)?
4. Can we reproduce the "multiple instances" claim with a simple two-object image?
