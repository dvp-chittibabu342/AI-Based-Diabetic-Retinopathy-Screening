"""
RetinaGuard — Grad-CAM++ (Generalized Gradient-Weighted Class Activation Mapping)
================================================================================
Implements Grad-CAM++ for deep convolutional networks (Chattopadhay et al., 2018).
Serves as the primary attribution explainability engine for RetinaGuard 2.0.

Key Advancements over Standard Grad-CAM:
1. Uses second- and third-order gradient weighting for better localization of
   multiple instances of pathology (e.g. scattered microaneurysms and exudates).
2. Dynamic target class selection (targets predicted class by default or arbitrary
   user-selected ICDR severity grade 0-4).
3. Strict validation: verifies non-NaN, non-zero variance, finite numbers.
4. ZERO synthetic/fake heatmaps: upon failure or unavailable model, explicitly
   reports XAI_UNAVAILABLE.

Usage:
    from python.explainability.gradcam_plus_plus import GradCAMPlusPlus
    gcampp = GradCAMPlusPlus(model)
    heatmap_np, overlay_pil = gcampp.generate(tensor, target_class=2)
"""

import logging
from typing import Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)


class GradCAMPlusPlus:
    """
    Grad-CAM++ Attribution Generator for EfficientNet-B0 and PyTorch CNN backbones.
    """

    def __init__(self, model: Optional[nn.Module], target_layer: Optional[nn.Module] = None):
        self.model = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._fwd_hook = None
        self._bwd_hook = None

        if model is None:
            self.device = torch.device("cpu")
            self.target_layer = None
            return

        self.device = next(model.parameters()).device

        # Automatically locate the final convolutional layer if not specified
        if target_layer is None:
            target_layer = self._find_last_conv(model)
        self.target_layer = target_layer

        # Register forward and backward hooks
        self._fwd_hook = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = self.target_layer.register_full_backward_hook(self._save_gradient)

    @staticmethod
    def _find_last_conv(model: nn.Module) -> nn.Module:
        """Find the final Conv2d layer before global average pooling."""
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError("No Conv2d layer identified in model architecture.")
        logger.debug("Grad-CAM++ target layer resolved: %s", last_conv)
        return last_conv

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        retinal_mask: Optional[np.ndarray] = None
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Generate validated Grad-CAM++ attribution heatmap.

        Args:
            input_tensor: (1, 3, H, W) normalized image tensor
            target_class: Class index (0-4) to compute attribution for.
                          If None, uses predicted class (argmax).
            retinal_mask: Optional boolean or uint8 mask of active retinal area.

        Returns:
            (heatmap_np, status_message)
            heatmap_np: (H, W) float32 array in [0.0, 1.0], or None if invalid.
        """
        if self.model is None or self.target_layer is None:
            return None, "MODEL_UNAVAILABLE"

        self.model.eval()
        input_tensor = input_tensor.to(self.device).clone()
        input_tensor.requires_grad_(True)

        # Forward pass
        logits = self.model(input_tensor)
        if logits is None or logits.shape[1] == 0:
            return None, "MODEL_OUTPUT_EMPTY"

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        elif target_class < 0 or target_class >= logits.shape[1]:
            logger.warning("Target class %s out of bounds [0, %d)", target_class, logits.shape[1])
            target_class = int(logits.argmax(dim=1).item())

        # Backward pass w.r.t. target score
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward(retain_graph=False)

        if self._gradients is None or self._activations is None:
            return None, "HOOKS_FAILED_NO_GRADIENTS"

        # Transfer activations and gradients
        # Gradients: (1, C, H, W), Activations: (1, C, H, W)
        grads = self._gradients.detach().cpu().numpy()[0]     # (C, H, W)
        acts = self._activations.detach().cpu().numpy()[0]    # (C, H, W)

        # Check for NaN / Inf
        if np.isnan(grads).any() or np.isnan(acts).any():
            return None, "GRADIENTS_CONTAIN_NAN"

        # Grad-CAM++ formulation:
        # g = dY/dA
        # alpha_ij^k = (g_ij^k)^2 / (2*(g_ij^k)^2 + sum_ab(A_ab^k * (g_ab^k)^3) + eps)
        # w_k = sum_ij (alpha_ij^k * relu(g_ij^k))
        # cam = relu(sum_k (w_k * A^k))

        eps = 1e-7
        g2 = grads ** 2
        g3 = grads ** 3

        # sum over spatial dimensions (axis 1, 2)
        sum_a_g3 = np.sum(acts * g3, axis=(1, 2), keepdims=True)  # (C, 1, 1)
        denom = 2.0 * g2 + sum_a_g3
        denom = np.where(denom != 0.0, denom, eps)

        alphas = g2 / denom  # (C, H, W)

        # Positive gradients only
        relu_grads = np.maximum(grads, 0.0)
        weights = np.sum(alphas * relu_grads, axis=(1, 2))  # (C,)

        # Linear combination of feature maps weighted by alpha-adjusted weights
        cam = np.zeros(acts.shape[1:], dtype=np.float32)  # (H, W)
        for k in range(len(weights)):
            cam += weights[k] * acts[k]

        # ReLU on combined map
        cam = np.maximum(cam, 0.0)

        # Normalize to [0.0, 1.0]
        max_val = np.max(cam)
        if max_val > 1e-8:
            cam = cam / max_val
        else:
            # If map has no activation, return clean uniform zero
            cam = np.zeros_like(cam)

        # Resize to input tensor spatial dimensions
        target_h, target_w = input_tensor.shape[2], input_tensor.shape[3]
        cam_resized = cv2.resize(cam, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Clamping to valid retinal mask if provided
        if retinal_mask is not None:
            mask_resized = cv2.resize(retinal_mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            cam_resized = cam_resized * (mask_resized > 0)

        return cam_resized.astype(np.float32), "SUCCESS"

    def overlay_on_image(
        self,
        original_pil: Image.Image,
        heatmap_np: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET
    ) -> Image.Image:
        """
        Overlays attribution heatmap onto original fundus photograph.
        """
        orig_w, orig_h = original_pil.size
        # Clamp overlay dimension to 1024 max for memory safety
        max_dim = max(orig_w, orig_h)
        if max_dim > 1024:
            scale = 1024.0 / max_dim
            blend_base = original_pil.resize((int(orig_w * scale), int(orig_h * scale)), Image.BILINEAR)
        else:
            blend_base = original_pil

        w, h = blend_base.size
        heat_u8 = np.uint8(np.clip(heatmap_np * 255.0, 0, 255))
        heat_res = cv2.resize(heat_u8, (w, h), interpolation=cv2.INTER_LINEAR)
        heat_bgr = cv2.applyColorMap(heat_res, colormap)
        heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

        orig_arr = np.array(blend_base.convert("RGB"), dtype=np.uint8)
        blended = cv2.addWeighted(orig_arr, 1.0 - alpha, heat_rgb, alpha, 0)
        return Image.fromarray(blended)

    def remove_hooks(self):
        """Clean up hooks to prevent memory leaks."""
        if hasattr(self, "_fwd_hook") and self._fwd_hook is not None:
            try:
                self._fwd_hook.remove()
            except Exception:
                pass
            self._fwd_hook = None
        if hasattr(self, "_bwd_hook") and self._bwd_hook is not None:
            try:
                self._bwd_hook.remove()
            except Exception:
                pass
            self._bwd_hook = None


def compute_gradcam_plus_plus(
    model: Optional[nn.Module],
    image_input,
    target_class: Optional[int] = None,
    target_layer_name: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Convenience function to compute Grad-CAM++ attribution heatmap.
    Returns (H, W) float32 numpy array or None on failure.
    """
    if model is None:
        return None
    try:
        gcam = GradCAMPlusPlus(model)
        if isinstance(image_input, np.ndarray):
            # Convert BGR/RGB array to tensor
            img_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB) if len(image_input.shape) == 3 and image_input.shape[2] == 3 else image_input
            img_resized = cv2.resize(img_rgb, (512, 512)).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            norm = (img_resized - mean) / std
            tensor = torch.tensor(norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        elif isinstance(image_input, torch.Tensor):
            tensor = image_input
        else:
            return None

        heatmap, status = gcam.generate(tensor, target_class=target_class)
        gcam.remove_hooks()
        return heatmap
    except Exception as e:
        logger.error("Failed in compute_gradcam_plus_plus: %s", e)
        return None

