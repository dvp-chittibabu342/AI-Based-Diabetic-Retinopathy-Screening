"""
grading_service.py
==================
Deep Learning DR Severity Grading & Grad-CAM++ Attribution Service.
Uses the locked production model: EXP-001 (EfficientNet-B0).

RetinaGuard 2.0 Compliance:
1. Complete 5-class ICDR probability distribution from softmax output.
2. Transparent uncertainty metrics: top probability, runner-up probability, and margin.
3. Explicit calibration status: "Calibration: not applied (raw model probabilities)".
4. Primary attribution: Grad-CAM++ (2nd and 3rd order gradients w.r.t. final Conv2d block).
5. Fallback attribution: Standard Grad-CAM.
6. Honest failure: If XAI is unavailable, returns status 'XAI_UNAVAILABLE' with NO synthetic images.
7. Dynamic target class selection (targets predicted grade by default).
8. Configured prototype screening rule: Grade >= 2 flagged as referable DR.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
from PIL import Image
import yaml
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "python"))

from inference.predict import DRPredictor, REFERRAL_TEXTS, REFERRAL_THRESHOLD
from explainability.gradcam_plus_plus import GradCAMPlusPlus
from explainability.gradcam import GradCAM

# Load verified class mapping from model configuration
CLASS_MAPPING_FILE = ROOT / "models" / "aptos_efficientnet" / "class_mapping.json"
if CLASS_MAPPING_FILE.exists():
    with open(CLASS_MAPPING_FILE, "r") as f:
        _mapping = json.load(f)
        CLASS_NAMES = [_mapping[str(i)] for i in range(5)]
else:
    CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]


class GradingService:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load project configuration
        config_path = ROOT / "python" / "config" / "config.yaml"
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.cfg["_root"] = str(ROOT)

        # Initialize DRPredictor using production model
        self.predictor = DRPredictor(self.cfg)
        self.model_info = "EXP-001 (EfficientNet-B0)"

        # Load production ONNX model via config-based relative path
        onnx_rel_path = self.cfg.get("paths", {}).get("onnx_model", "models/aptos_efficientnet/best_model.onnx")
        self.onnx_path = ROOT / onnx_rel_path
        self.onnx_session = None
        self.onnx_input_name = None
        if self.onnx_path.exists():
            try:
                import onnxruntime as ort
                self.onnx_session = ort.InferenceSession(
                    str(self.onnx_path),
                    providers=["CPUExecutionProvider"]
                )
                self.onnx_input_name = self.onnx_session.get_inputs()[0].name
            except Exception:
                self.onnx_session = None

    def grade(
        self,
        pil_image: Image.Image,
        target_class: Optional[int] = None,
        filename_prefix: str = "attribution"
    ) -> Dict[str, Any]:
        """
        Runs deep learning inference and generates Grad-CAM++ attribution heatmap.
        Uses production ONNX model (EXP-001) with PyTorch fallback.
        """
        # 1. Forward pass (Primary: ONNX runtime, Fallback: PyTorch)
        tensor = self.predictor.preprocessor.preprocess_for_inference(pil_image)
        if self.onnx_session is not None:
            ort_inputs = {self.onnx_input_name: tensor.cpu().numpy()}
            logits = self.onnx_session.run(None, ort_inputs)[0]
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = (exp_logits / np.sum(exp_logits, axis=1, keepdims=True)).squeeze()
        else:
            self.predictor.model.eval()
            with torch.no_grad():
                logits = self.predictor.model(tensor.to(self.predictor.device))
                probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

        grade = int(np.argmax(probs))
        confidence = float(probs[grade])
        is_referable = bool(grade >= REFERRAL_THRESHOLD)
        conf_level = "High" if confidence >= 0.80 else ("Moderate" if confidence >= 0.50 else "Borderline")

        # Select target class for attribution (defaults to predicted grade)
        actual_target_class = target_class if target_class is not None else grade
        actual_target_class = max(0, min(4, int(actual_target_class)))

        # Mathematical derivation of transparent uncertainty indicators
        sorted_indices = np.argsort(probs)[::-1]
        top_idx = int(sorted_indices[0])
        runner_up_idx = int(sorted_indices[1])
        top_p = float(probs[top_idx])
        runner_up_p = float(probs[runner_up_idx])
        margin = float(top_p - runner_up_p)

        # 2. XAI Attribution: Grad-CAM++ (Primary) with Grad-CAM (Fallback)
        gcam_url = None
        xai_method = "Grad-CAM++"
        xai_status = "NOT_RUN"
        target_layer_desc = "backbone.features.8.0 (final Conv2d, 1280 channels)"

        if self.predictor.model is not None:
            try:
                # Primary: Grad-CAM++
                gcampp = GradCAMPlusPlus(self.predictor.model)
                gcam_tensor = self.predictor.preprocessor.preprocess_for_inference(pil_image)
                heatmap, status = gcampp.generate(gcam_tensor, target_class=actual_target_class)
                
                if heatmap is not None and status == "SUCCESS":
                    overlay = gcampp.overlay_on_image(pil_image, heatmap, alpha=0.45)
                    gcam_filename = f"{filename_prefix}_gcampp_{int(np.random.randint(100000, 999999))}.png"
                    gcam_path = self.output_dir / gcam_filename
                    overlay.save(gcam_path)
                    gcam_url = f"/outputs/{gcam_filename}"
                    xai_status = "SUCCESS"
                    xai_method = "Grad-CAM++"
                gcampp.remove_hooks()
            except Exception as e:
                # Fallback: standard Grad-CAM
                try:
                    gcam = GradCAM(self.predictor.model)
                    gcam_tensor = self.predictor.preprocessor.preprocess_for_inference(pil_image)
                    heatmap, _ = gcam.generate(gcam_tensor, target_class=actual_target_class)
                    if heatmap is not None and not np.isnan(heatmap).any():
                        overlay = gcam.overlay_on_image(pil_image, heatmap, alpha=0.45)
                        gcam_filename = f"{filename_prefix}_gcam_{int(np.random.randint(100000, 999999))}.png"
                        gcam_path = self.output_dir / gcam_filename
                        overlay.save(gcam_path)
                        gcam_url = f"/outputs/{gcam_filename}"
                        xai_status = "SUCCESS"
                        xai_method = "Grad-CAM (Fallback)"
                    gcam.remove_hooks()
                except Exception:
                    gcam_url = None
                    xai_status = "XAI_UNAVAILABLE"
        else:
            gcam_url = None
            xai_status = "XAI_UNAVAILABLE"

        # Build clean probabilities mapping
        class_probs = []
        for idx, p in enumerate(probs):
            class_probs.append({
                "grade": idx,
                "label": CLASS_NAMES[idx],
                "probability": round(float(p), 4),
                "percentage": round(float(p) * 100.0, 1)
            })

        return {
            "grade": grade,
            "severity_name": CLASS_NAMES[grade],
            "confidence": round(confidence, 4),
            "confidence_percent": round(confidence * 100.0, 1),
            "confidence_level": conf_level,
            "probabilities": class_probs,
            "top_probability": round(top_p, 4),
            "runner_up_grade": runner_up_idx,
            "runner_up_name": CLASS_NAMES[runner_up_idx],
            "runner_up_probability": round(runner_up_p, 4),
            "margin": round(margin, 4),
            "margin_percent": round(margin * 100.0, 1),
            "decision_margin_percent": round(margin * 100.0, 1),
            "calibration_status": "Calibration: not applied (raw model probabilities)",
            "referral": is_referable,
            "referral_badge": "REFERABLE DR" if is_referable else "NON-REFERABLE",
            "referral_rule": "Configured prototype screening rule (Grade >= 2)",
            "referral_action": "Ophthalmology Referral Required" if is_referable else "Routine Follow-up Advised",
            "recommendation": REFERRAL_TEXTS.get(grade, "Routine clinical evaluation."),
            "model_version": "EXP-001",
            "architecture": "EfficientNet-B0 (Focal Loss, Balanced Sampling)",
            "gradcam_url": gcam_url,
            "xai": {
                "method": xai_method,
                "status": xai_status,
                "target_class": actual_target_class,
                "target_class_name": CLASS_NAMES[actual_target_class],
                "target_layer": target_layer_desc,
                "gradcam_url": gcam_url,
                "notice": "Attribution visualization shows regions contributing to model prediction; it is not confirmed lesion segmentation."
            },
            "gradcam_disclaimer": "Model Attribution Map: Highlights spatial regions that contributed most strongly to the neural network prediction. This is an explainability tool, not a manual lesion segmentation."
        }
