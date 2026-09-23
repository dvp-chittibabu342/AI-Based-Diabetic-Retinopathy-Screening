"""
RetinaGuard — Region Sensitivity Analysis
==========================================
Implements on-demand region masking analysis to measure model sensitivity
to candidate lesion regions without altering base model weights.

Strict Neutral Terminology:
- Described as: "Model sensitivity to region masking"
- Never described as causal proof or manual lesion validation.
- Output delta: "Masking this region changed the selected class probability by X percentage points."
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class RegionSensitivityAnalyzer:
    """
    Evaluates how masking candidate evidence regions impacts EXP-001 class probabilities.
    """

    def __init__(self, model: torch.nn.Module, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.device = next(model.parameters()).device

    def analyze_regions(
        self,
        image_pil: Image.Image,
        regions: List[Dict[str, Any]],
        target_class: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Runs on-demand perturbation analysis for up to 3 candidate evidence regions.

        Args:
            image_pil: Original PIL retinal fundus image
            regions: List of up to 3 dicts containing 'bbox' [x, y, w, h] or 'label'
            target_class: Target ICDR severity class (0-4). Defaults to model argmax.

        Returns:
            Structured dict conforming to AnalysisResult['region_sensitivity']
        """
        # 1. Base inference on original unmasked image
        base_tensor = self.preprocessor.preprocess_for_inference(image_pil).to(self.device)
        self.model.eval()
        with torch.no_grad():
            base_logits = self.model(base_tensor)
            base_probs = F.softmax(base_logits, dim=1).squeeze().cpu().numpy()

        if target_class is None:
            target_class = int(np.argmax(base_probs))
        else:
            target_class = max(0, min(4, int(target_class)))

        orig_prob = float(base_probs[target_class])
        orig_pct = orig_prob * 100.0

        # Limit to 3 regions max
        capped_regions = regions[:3] if regions else []
        tested_results = []

        img_np = np.array(image_pil.convert("RGB"))
        h, w = img_np.shape[:2]

        for idx, reg in enumerate(capped_regions):
            reg_id = reg.get("id", f"region_{idx + 1}")
            reg_label = reg.get("label", f"Candidate Region {idx + 1}")
            bbox = reg.get("bbox", None)

            # Determine mask bounding box
            if bbox is not None and len(bbox) == 4:
                bx, by, bw, bh = [int(v) for v in bbox]
                x0, y0 = max(0, bx), max(0, by)
                x1, y1 = min(w, bx + bw), min(h, by + bh)
            else:
                # Default centered region probe if no bbox provided
                cx, cy = w // 2, h // 2
                rad = max(10, min(w, h) // 16)
                x0, y0 = max(0, cx - rad), max(0, cy - rad)
                x1, y1 = min(w, cx + rad), min(h, cy + rad)

            # Create masked copy
            masked_np = img_np.copy()
            # Replace target region with median background of outer border margin
            outer_y0, outer_y1 = max(0, y0 - 15), min(h, y1 + 15)
            outer_x0, outer_x1 = max(0, x0 - 15), min(w, x1 + 15)
            surrounding = img_np[outer_y0:outer_y1, outer_x0:outer_x1]
            if surrounding.size > 0:
                fill_color = np.median(surrounding, axis=(0, 1)).astype(np.uint8)
            else:
                fill_color = np.array([120, 60, 20], dtype=np.uint8)

            masked_np[y0:y1, x0:x1] = fill_color
            masked_pil = Image.fromarray(masked_np)

            # Run inference on masked image
            masked_tensor = self.preprocessor.preprocess_for_inference(masked_pil).to(self.device)
            with torch.no_grad():
                masked_logits = self.model(masked_tensor)
                masked_probs = F.softmax(masked_logits, dim=1).squeeze().cpu().numpy()

            masked_prob = float(masked_probs[target_class])
            masked_pct = masked_prob * 100.0
            delta_prob = round(masked_prob - orig_prob, 4)
            delta_pts = round(masked_pct - orig_pct, 2)

            if delta_pts < -0.1:
                summary = f"Masking this region reduced the selected class probability by {abs(delta_pts):.1f} percentage points."
            elif delta_pts > 0.1:
                summary = f"Masking this region increased the selected class probability by {abs(delta_pts):.1f} percentage points."
            else:
                summary = "Masking this region produced no measurable change in the selected class probability."

            tested_results.append({
                "region_id": reg_id,
                "label": reg_label,
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "masked_probability": round(masked_prob, 4),
                "delta_prob": delta_prob,
                "delta_percentage_points": delta_pts,
                "summary": summary
            })

        results_formatted = []
        for r in tested_results:
            results_formatted.append({
                "region_id": r["region_id"],
                "region_name": r.get("label", r["region_id"]),
                "region_type": "Candidate Region",
                "target_class": target_class,
                "original_prob": round(orig_prob, 4),
                "masked_prob": r["masked_probability"],
                "prob_delta": r["delta_prob"],
                "delta_percent": r["delta_percentage_points"],
                "statement": f"Model sensitivity to region masking: {r['summary']}"
            })

        return {
            "method": "Model sensitivity to region masking",
            "target_class": target_class,
            "original_probability": round(orig_prob, 4),
            "original_percentage": round(orig_pct, 2),
            "regions_tested_count": len(tested_results),
            "regions": tested_results,
            "results": results_formatted,
            "overall_interpretation": "Evaluates change in predicted probability when candidate evidence is masked. Not causal proof.",
            "disclaimer": "Model sensitivity to region masking measures local perturbation response. It is not causal proof or manual lesion validation."
        }


    def analyze(self, image_input, candidate_regions: List[Dict[str, Any]], target_class: Optional[int] = None) -> Dict[str, Any]:
        """Convenience alias accepting either PIL Image or numpy array."""
        if isinstance(image_input, np.ndarray):
            import cv2
            img_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB) if len(image_input.shape) == 3 and image_input.shape[2] == 3 else image_input
            image_pil = Image.fromarray(img_rgb)
        else:
            image_pil = image_input

        res = self.analyze_regions(image_pil, candidate_regions, target_class=target_class)
        res["results"] = []
        for r in res["regions"]:
            res["results"].append({
                "region_id": r["region_id"],
                "region_name": r.get("label", r["region_id"]),
                "region_type": "Candidate Region",
                "target_class": res["target_class"],
                "original_prob": res["original_probability"],
                "masked_prob": r["masked_probability"],
                "prob_delta": r["delta_prob"],
                "delta_percent": r["delta_percentage_points"],
                "statement": f"Model sensitivity to region masking: {r['summary']}"
            })
        return res

