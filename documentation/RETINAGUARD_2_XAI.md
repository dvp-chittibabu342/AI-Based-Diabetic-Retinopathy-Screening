# RetinaGuard 2.0 — Explainability (XAI) & Attribution Specification
**SIH 2026 Problem Statement: SIH26038**  
**Team: VyuhaX**  
**Module: `python/explainability/`**  

---

## 1. Executive Summary

In clinical artificial intelligence, "explainability" is frequently reduced to decorative visualizations or unverified heatmaps that mislead clinicians into believing a neural network is segmenting microaneurysms or exudates. RetinaGuard 2.0 takes an uncompromising stance on scientific and clinical transparency:

1. **Model Attribution, Not Lesion Segmentation**: Saliency heatmaps indicate which receptive fields in the image influenced the neural network's final logit score. They do **not** represent histological or confirmed anatomical lesion boundaries.
2. **Real Grad-CAM++ Implementation**: Uses second- and third-order gradient weighting (Chattopadhay et al., 2018) to improve spatial localization of scattered microvascular abnormalities across multiple retinal quadrants.
3. **Zero Synthetic Heatmap Policy**: All synthetic Gaussian blobs, fake radial gradients, and `_placeholder_heatmap` fallbacks have been completely excised. If gradient backpropagation is unavailable, the system explicitly returns `XAI_UNAVAILABLE`.
4. **Dynamic Target Class Selection**: Healthcare screeners can dynamically evaluate attribution not only for the model's top predicted grade, but across any alternative ICDR grade (0 to 4) to investigate class counter-evidence.
5. **On-Demand Region Sensitivity Masking**: Allows screeners to temporarily obscure candidate lesion regions and re-execute EXP-001 forward inference, reporting probability deltas with strict neutral framing: *"Model sensitivity to region masking"*.

---

## 2. Mathematical Formulation of Grad-CAM++

Standard Grad-CAM computes feature map weights $w_k^c$ via global average pooling of first-order gradients:

$$w_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial Y^c}{\partial A_{i,j}^k}$$

While effective for coarse semantic objects (e.g., classifying dogs vs cats), standard Grad-CAM exhibits severe limitations in medical imaging:
- When multiple instances of small pathology (e.g. 10 scattered microaneurysms) appear in the same scan, gradients with opposite signs or differing magnitudes wash out across the feature map.
- The spatial localization of sub-millimeter retinal lesions becomes blurred.

### The Grad-CAM++ Formulation

Grad-CAM++ reformulates the weighting coefficient $w_k^c$ as a weighted sum of positive gradients, incorporating higher-order derivatives ($g^2$ and $g^3$):

$$w_k^c = \sum_{i} \sum_{j} \alpha_{i,j}^{k,c} \cdot \text{ReLU}\left(\frac{\partial Y^c}{\partial A_{i,j}^k}\right)$$

Where the analytical weighting coefficients $\alpha_{i,j}^{k,c}$ are derived from second- and third-order partial derivatives of the class score $Y^c$ with respect to activation map $A^k$:

$$\alpha_{i,j}^{k,c} = \frac{\left(\frac{\partial Y^c}{\partial A_{i,j}^k}\right)^2}{2\left(\frac{\partial Y^c}{\partial A_{i,j}^k}\right)^2 + \sum_{a} \sum_{b} A_{a,b}^k \left(\frac{\partial Y^c}{\partial A_{i,j}^k}\right)^3 + \epsilon}$$

Where:
- $Y^c$ is the unnormalized class score (logit) for ICDR severity grade $c \in \{0, 1, 2, 3, 4\}$.
- $A_{i,j}^k$ is the spatial activation at position $(i, j)$ in channel $k$ of the target layer.
- $\epsilon = 10^{-7}$ prevents numerical division by zero.

The resulting class-discriminative saliency map $L_{\text{Grad-CAM++}}^c$ is computed by taking the positive linear combination:

$$L_{\text{Grad-CAM++}}^c = \text{ReLU}\left(\sum_{k} w_k^c A^k\right)$$

The map is subsequently normalized to $[0.0, 1.0]$ and upsampled via bilinear interpolation to match the input fundus image dimensions.

---

## 3. Target Layer Specification in EfficientNet-B0

In RetinaGuard 2.0's EXP-001 backbone (EfficientNet-B0), Grad-CAM++ attaches forward and backward hooks to:
- **Module Path**: `backbone.features.8.0` (or `features.8.0`)
- **Layer Type**: `nn.Conv2d`
- **Output Channels**: 1280
- **Spatial Resolution**: $7 \times 7$ feature grid (at $224 \times 224$ input resolution)

This layer represents the highest-level spatial representation before global average pooling (`AdaptiveAvgPool2d`) and the final linear classification head.

---

## 4. Policy on Synthetic Heatmaps & Honest Refusal

Prior versions or simplistic AI demos sometimes utilized heuristic "fake heatmaps" (e.g. generating a Gaussian circle over the image center or fovea) if gradient computation encountered errors or if PyTorch was not initialized.

**RetinaGuard 2.0 completely prohibits fabricated visual heatmaps.**

### Enforcement in Code:
1. `_placeholder_heatmap` has been permanently deleted from `python/explainability/gradcam.py`.
2. If the PyTorch model graph or backward hooks are unavailable:
   - Saliency generation returns `(None, "MODEL_UNAVAILABLE")`.
   - The UI disables the attribution layer toggle.
   - The status is recorded explicitly in the contract as `"status": "XAI_UNAVAILABLE"`.
   - No blended image is displayed or saved.

---

## 5. Dynamic Target Class Selection

Most medical AI applications only display attribution for the model's top predicted class ($\arg\max_c Y^c$). However, differential screening requires investigating:
- *"Why did the model assign 23% probability to Mild DR even though it predicted Moderate DR?"*
- *"Does the model see any features supporting Proliferative DR?"*

RetinaGuard 2.0 implements interactive dynamic class targeting:
1. By default, attribution targets the predicted grade ($c = \text{grade}$).
2. Screeners can select any of the 5 ICDR classes from the dropdown:
   - **Grade 0**: No DR
   - **Grade 1**: Mild DR
   - **Grade 2**: Moderate DR
   - **Grade 3**: Severe DR
   - **Grade 4**: Proliferative DR
3. The backward pass backpropagates gradients specifically from logit $Y^{\text{target}}$, producing class-specific attribution maps.

---

## 6. On-Demand Region Sensitivity Masking

To test whether the model is truly sensitive to candidate lesion regions or merely relying on background retinal textures, RetinaGuard 2.0 provides on-demand perturbation analysis in `python/explainability/region_sensitivity.py`.

### Protocol:
1. Up to 3 candidate evidence regions (e.g. detected microaneurysms or hard exudates) are selected.
2. The original model probability $P(\text{class} = c)$ is computed on the unaltered scan.
3. For each candidate region, the pixel area defined by its bounding box $[x, y, w, h]$ is temporarily obscured using the median chromaticity of the surrounding retinal margin.
4. The **same EXP-001 model weights** are executed in a forward pass on the masked image to obtain perturbed probability $P_{\text{masked}}(\text{class} = c)$.
5. The sensitivity delta is calculated:

$$\Delta P = P_{\text{masked}} - P_{\text{original}}$$

$$\Delta\% = (P_{\text{masked}} \times 100) - (P_{\text{original}} \times 100)$$

### Neutral Terminology Requirements:
- Always labeled: **"Model sensitivity to region masking"**
- Statement format: *"Masking this region changed the selected class probability by X percentage points."*
- **Never labeled**: "Causal proof", "lesion validation", or "proven diagnostic pathology".

---

## 7. Comparative Summary: Attribution vs Evidence

| Aspect | Grad-CAM++ Attribution | Candidate Lesion Evidence |
| :--- | :--- | :--- |
| **Source** | 1280-channel feature gradients from EXP-001 | U-Net / Morphological computer vision filters |
| **Nature** | Model decision localization | Visual pathology candidates |
| **Output Type** | Continuous $[0, 1]$ saliency heatmap | Candidate counts, bounding boxes, $96 \times 96$ crops |
| **Clinical Meaning** | *"Where the model looked to decide"* | *"Morphological candidates identified for human review"* |
| **Diagnostic Proof** | **Not** causal proof or confirmed lesion segmentation | **Not** confirmed clinical diagnoses |
| **Failure State** | Explicit `XAI_UNAVAILABLE` refusal | Classical CV fallback mode |
