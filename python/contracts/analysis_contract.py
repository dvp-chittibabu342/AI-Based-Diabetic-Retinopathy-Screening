"""
RetinaGuard — Central Analysis Result Contract
==============================================
The single unified source of truth for RetinaGuard 2.0.
All values rendered in the UI, API responses, ExplainAI engine, and HTML reports
originate strictly from this structured schema.

Guarantees:
- No independently calculated or diverging values for predictions, probabilities,
  quality metrics, candidate counts, XAI attribution, or explanations.
- Complete 5-class probability vector with top probability, runner-up, and margin.
- Transparent calibrated status labeling: "Calibration: not applied".
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent

# Load verified class mapping from model configuration
CLASS_MAPPING_FILE = ROOT / "models" / "aptos_efficientnet" / "class_mapping.json"
if CLASS_MAPPING_FILE.exists():
    with open(CLASS_MAPPING_FILE, "r") as f:
        CLASS_NAMES = json.load(f)
        # Ensure integer keys
        CLASS_NAMES = {int(k): str(v) for k, v in CLASS_NAMES.items()}
else:
    CLASS_NAMES = {
        0: "No DR",
        1: "Mild DR",
        2: "Moderate DR",
        3: "Severe DR",
        4: "Proliferative DR"
    }

SEVERITY_SCALE = [
    {"grade": 0, "label": CLASS_NAMES.get(0, "No DR"), "description": "No visible microvascular lesions."},
    {"grade": 1, "label": CLASS_NAMES.get(1, "Mild DR"), "description": "Microaneurysms only."},
    {"grade": 2, "label": CLASS_NAMES.get(2, "Moderate DR"), "description": "More than microaneurysms, but less than severe NPDR (exudates, mild hemorrhages)."},
    {"grade": 3, "label": CLASS_NAMES.get(3, "Severe DR"), "description": "Extensive intra-retinal hemorrhages, venous beading, or prominent IRMA."},
    {"grade": 4, "label": CLASS_NAMES.get(4, "Proliferative DR"), "description": "Neovascularization, vitreous/preretinal hemorrhage."}
]


def create_empty_analysis_result(
    patient_id: str = "PT-82910",
    exam_id: str = "EX-00412",
    eye: str = "Right (OD)",
    original_url: str = ""
) -> Dict[str, Any]:
    """Factory creating an empty, standardized AnalysisResult instance."""
    now = datetime.now()
    analysis_id = f"RG2-{now.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"

    return {
        "contract_version": "2.0.0",
        "metadata": {
            "analysis_id": analysis_id,
            "timestamp": now.isoformat(),
            "formatted_time": now.strftime("%d-%b-%Y %H:%M:%S"),
            "pipeline_version": "2.0.0",
            "model_version": "EXP-001",
            "architecture": "EfficientNet-B0 (Focal Loss, Balanced Sampling)",
            "device": "CPU (Offline Edge Runtime)",
            "image_hash": ""
        },
        "input": {
            "patient_id": patient_id,
            "exam_id": exam_id,
            "eye": eye,
            "width": 0,
            "height": 0,
            "original_url": original_url,
            "demo_reference": None
        },
        "quality": {
            "score": 0.0,
            "status": "UNGRADABLE",
            "gradable": False,
            "focus": "Poor",
            "illumination": "Poor",
            "contrast": "Poor",
            "field_of_view": "Poor",
            "snr_db": 0.0,
            "laplacian_var": 0.0,
            "mean_lum": 0.0,
            "fov_percent": 0.0,
            "reason": "Analysis pending.",
            "recommendation": "Recapture retinal image with valid camera output.",
            "threshold_notes": "Configured prototype thresholds: minimum composite score 40.0, sharpness variance >= 35.0, illumination in [30.0, 225.0]."
        },
        "preprocessing": {
            "steps_applied": [],
            "crop_bbox": None,
            "enhanced_url": "",
            "resolution_standardized": [512, 512]
        },
        "anatomy": {
            "optic_disc": {
                "detected": False,
                "centroid": [0, 0],
                "radius_px": 0,
                "methodology": "Computer Vision Peak Luminance & Morphology"
            },
            "fovea": {
                "detected": False,
                "coordinates": [0, 0],
                "radius_px": 0,
                "methodology": "Anatomical Temporal Geometric Projection"
            },
            "vessels": {
                "detected": False,
                "density_percent": 0.0,
                "methodology": "Multi-scale Morphological Black Top-Hat & Ridge Filtering"
            },
            "vessel_overlay_url": "",
            "landmarks_overlay_url": ""
        },
        "lesion_candidates": {
            "mode": "CANDIDATE DETECTION",
            "badge_label": "CANDIDATE EVIDENCE",
            "is_ai": False,
            "disclaimer": "Candidate evidence regions identified by computer vision; not independently confirmed clinical lesions.",
            "microaneurysms": {
                "candidate_count": 0,
                "total_area_px": 0,
                "label": "Microaneurysm candidates",
                "methodology": "Green-channel morphological top-hat filtering / U-Net"
            },
            "hard_exudates": {
                "candidate_count": 0,
                "total_area_px": 0,
                "label": "Hard exudate candidates",
                "methodology": "High-intensity clustering excluding optic disc mask"
            },
            "hemorrhages": {
                "candidate_count": 0,
                "total_area_px": 0,
                "label": "Hemorrhage candidates",
                "methodology": "Dark intra-retinal lesion segmentation outside vessel tree"
            },
            "soft_exudates": {
                "candidate_count": 0,
                "total_area_px": 0,
                "label": "Soft exudate candidates (Cotton wool spots)",
                "methodology": "Faint fuzzy opacity clustering"
            },
            "neovascularization": {
                "risk_level": "Low",
                "label": "Neovascularization Risk Indicator",
                "description": "Vascular density within expected range.",
                "methodology": "Vascular density stratification heuristic indicator (not clinical diagnosis)"
            },
            "candidate_crops": [],
            "overlay_url": ""
        },
        "model": {
            "predicted_grade": 0,
            "severity_name": CLASS_NAMES.get(0, "No DR"),
            "confidence": 0.0,
            "confidence_percent": 0.0,
            "confidence_level": "Borderline",
            "probabilities": [],
            "top_probability": 0.0,
            "runner_up_grade": None,
            "runner_up_name": "N/A",
            "runner_up_probability": 0.0,
            "margin": 0.0,
            "margin_percent": 0.0,
            "calibration_status": "Calibration: not applied (raw model probabilities)",
            "referral": False,
            "referral_badge": "NON-REFERABLE",
            "referral_rule": "Configured prototype screening rule (Grade >= 2)",
            "recommendation": "Routine follow-up in 12 months.",
            "severity_progression": SEVERITY_SCALE
        },
        "xai": {
            "method": "Grad-CAM++",
            "status": "NOT_RUN",
            "target_class": 0,
            "target_class_name": CLASS_NAMES.get(0, "No DR"),
            "target_layer": "backbone.features.8.0 (final Conv2d, 1280 channels)",
            "gradcam_url": "",
            "notice": "Attribution visualization shows regions contributing to model prediction; it is not confirmed lesion segmentation."
        },
        "region_sensitivity": {
            "method": "Model sensitivity to region masking",
            "target_class": 0,
            "original_probability": 0.0,
            "original_percentage": 0.0,
            "regions_tested_count": 0,
            "regions": [],
            "disclaimer": "Model sensitivity to region masking measures local perturbation response. It is not causal proof or manual lesion validation."
        },
        "explanation": {
            "engine": "ExplainAI (Deterministic, Evidence-Grounded)",
            "questions": []
        },
        "screening": {
            "screening_status": "PENDING",
            "safety_gate_triggered": False,
            "referral_urgency": "Routine",
            "action_required": "Proceed with screening.",
            "disclaimer": "AI-assisted screening prototype. Not an autonomous clinical diagnosis. Confirm all findings with a licensed ophthalmologist."
        },
        "timings": {
            "quality_ms": 0.0,
            "enhancement_ms": 0.0,
            "anatomy_ms": 0.0,
            "lesions_ms": 0.0,
            "grading_ms": 0.0,
            "xai_ms": 0.0,
            "report_ms": 0.0,
            "total_ms": 0.0
        },
        "warnings": []
    }
