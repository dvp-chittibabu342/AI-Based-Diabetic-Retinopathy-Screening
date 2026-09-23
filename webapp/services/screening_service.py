"""
screening_service.py
====================
Master Orchestrator for RetinaGuard 2.0 Screening Workstation.
Coordinates Quality Gate, Filtering Transparency, Landmarks, Lesion Candidates,
EXP-001 Inference, Grad-CAM++ Attribution, Region Sensitivity, Grounded ExplainAI,
and Clinical Screening Reporting into the single unified AnalysisResult contract.
"""

import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import cv2
from PIL import Image

from webapp.services.quality_service import QualityService
from webapp.services.enhancement_service import EnhancementService
from webapp.services.structure_service import StructureService
from webapp.services.lesion_service import LesionService
from webapp.services.grading_service import GradingService
from webapp.services.report_service import ReportService

from python.contracts.analysis_contract import create_empty_analysis_result, CLASS_NAMES, SEVERITY_SCALE
from python.explainability.explain_ai import ExplainAIEngine
from python.explainability.region_sensitivity import RegionSensitivityAnalyzer

ROOT = Path(__file__).resolve().parent.parent.parent


class ScreeningService:
    def __init__(self, uploads_dir: Path, outputs_dir: Path):
        self.uploads_dir = uploads_dir
        self.outputs_dir = outputs_dir
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        # Initialize sub-services
        self.quality_service = QualityService()
        self.enhancement_service = EnhancementService(outputs_dir)
        self.structure_service = StructureService(outputs_dir)
        self.lesion_service = LesionService(outputs_dir)
        self.grading_service = GradingService(outputs_dir)
        self.report_service = ReportService(outputs_dir)

        # Demo cases directory
        self.demo_dir = ROOT / "demo" / "sample_images"

        # Region sensitivity analyzer instance
        self.region_analyzer = None
        if self.grading_service.predictor.model is not None:
            try:
                self.region_analyzer = RegionSensitivityAnalyzer(
                    self.grading_service.predictor.model,
                    self.grading_service.predictor.preprocessor
                )
            except Exception as e:
                print(f"[ScreeningService] RegionSensitivityAnalyzer notice: {e}")

    def get_demo_cases(self, include_all: bool = False) -> List[Dict[str, Any]]:
        """
        Returns curated authentic demo screening cases.
        If include_all is False, returns the 5 primary ICDR severity cases (Grades 0 to 4).
        If include_all is True, returns all 6 cases including Case 6 (Ungradable).
        """
        all_cases = [
            {
                "id": "grade_0",
                "label": "Case 1 — Normal Retina (No DR)",
                "reference_grade": 0,
                "reference_name": "No DR",
                "filename": "demo_grade0.png",
                "image_url": "/demo/sample_images/demo_grade0.png",
                "description": "Authentic normal retinal fundus with intact vasculature, clear macula, and no visible microvascular lesions."
            },
            {
                "id": "grade_1",
                "label": "Case 2 — Mild NPDR",
                "reference_grade": 1,
                "reference_name": "Mild NPDR",
                "filename": "demo_grade1.png",
                "image_url": "/demo/sample_images/demo_grade1.png",
                "description": "Authentic early-stage NPDR with microaneurysms. Clinical triage indicates non-referable observation."
            },
            {
                "id": "grade_2",
                "label": "Case 3 — Moderate NPDR",
                "reference_grade": 2,
                "reference_name": "Moderate NPDR",
                "filename": "demo_grade2.png",
                "image_url": "/demo/sample_images/demo_grade2.png",
                "description": "Authentic moderate NPDR with hard exudates and microvascular changes. Reaches referable DR threshold."
            },
            {
                "id": "grade_3",
                "label": "Case 4 — Severe NPDR",
                "reference_grade": 3,
                "reference_name": "Severe NPDR",
                "filename": "demo_grade3.png",
                "image_url": "/demo/sample_images/demo_grade3.png",
                "description": "Authentic severe NPDR exhibiting marked intra-retinal hemorrhages and vascular anomalies. Prompt specialist referral."
            },
            {
                "id": "grade_4",
                "label": "Case 5 — Proliferative DR",
                "reference_grade": 4,
                "reference_name": "Proliferative DR",
                "filename": "demo_grade4.png",
                "image_url": "/demo/sample_images/demo_grade4.png",
                "description": "Authentic proliferative diabetic retinopathy with advanced pathology and high neovascularization risk. Specialist referral required."
            },
            {
                "id": "case_6",
                "label": "Case 6 — Poor Quality (Ungradable)",
                "reference_grade": -1,
                "reference_name": "Ungradable / Recapture",
                "filename": "demo_case6_ungradable.png",
                "image_url": "/demo/sample_images/demo_case6_ungradable.png",
                "description": "Severe blur and illumination falloff simulating camera misfocus or poor pupil dilation. Demonstrates automated quality gate safety rejection."
            }
        ]
        valid_cases = [c for c in all_cases if (self.demo_dir / c["filename"]).is_file()]
        if not include_all:
            return [c for c in valid_cases if c["reference_grade"] >= 0]
        return valid_cases

    def run_screening(
        self,
        image_path: Path,
        original_url: str,
        patient_id: str = "PT-82910",
        exam_id: str = "EX-00412",
        eye: str = "Right (OD)",
        demo_reference: Optional[Dict[str, Any]] = None,
        target_class: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Runs the full RetinaGuard 2.0 automated screening pipeline.
        Returns the unified AnalysisResult object.
        """
        t0_total = time.perf_counter()

        # Load image via OpenCV and PIL
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Unable to read image at {image_path}. File may be corrupted.")

        pil_img = Image.open(image_path).convert("RGB")
        h, w = img_bgr.shape[:2]

        # Calculate file hash for auditability
        with open(image_path, "rb") as f:
            img_hash = hashlib.sha256(f.read()).hexdigest()[:16]

        # Instantiate unified AnalysisResult contract
        result = create_empty_analysis_result(
            patient_id=patient_id,
            exam_id=exam_id,
            eye=eye,
            original_url=original_url
        )
        result["metadata"]["image_hash"] = img_hash
        result["input"]["width"] = w
        result["input"]["height"] = h
        result["input"]["demo_reference"] = demo_reference

        # Top-level backward compatibility fields
        result["patient_id"] = patient_id
        result["exam_id"] = exam_id
        result["eye"] = eye
        result["demo_reference"] = demo_reference
        result["image_info"] = {
            "width": w,
            "height": h,
            "original_url": original_url
        }
        result["original_url"] = original_url

        # ---------------------------------------------------------------------
        # STAGE 1: IMAGE QUALITY ASSESSMENT GATE
        # ---------------------------------------------------------------------
        t0_q = time.perf_counter()
        quality = self.quality_service.assess(img_bgr)
        t_q = (time.perf_counter() - t0_q) * 1000.0
        result["timings"]["quality_ms"] = round(t_q, 1)
        result["quality"] = quality

        # Check for non-fatal quality warnings
        if quality.get("status") == "BORDERLINE":
            result["warnings"].append("Borderline image quality: subtle lesions may be difficult to discern.")

        # ---------------------------------------------------------------------
        # CRITICAL SAFETY GATE: UNGRADABLE REJECTION
        # ---------------------------------------------------------------------
        if not quality["gradable"]:
            result["screening"]["screening_status"] = "UNGRADABLE"
            result["screening"]["safety_gate_triggered"] = True
            result["screening"]["action_required"] = quality["recommendation"]
            result["screening"]["referral_urgency"] = "Recapture Required"

            # Reflect halted grading in central contract
            result["model"]["predicted_grade"] = -1
            result["model"]["severity_name"] = "UNGRADABLE"
            result["model"]["confidence"] = 0.0
            result["model"]["confidence_percent"] = 0.0
            result["model"]["confidence_level"] = "None"
            result["model"]["probabilities"] = []
            result["model"]["top_probability"] = 0.0
            result["model"]["referral"] = False
            result["model"]["referral_badge"] = "UNGRADABLE"
            result["model"]["referral_action"] = "Recapture Required"
            result["model"]["recommendation"] = quality["recommendation"]

            # Generate grounded explanations for ungradable rejection
            result["explanation"] = ExplainAIEngine.explain(result)

            # Generate ungradable clinical report
            t0_rep = time.perf_counter()
            report_res = self.report_service.generate(result, patient_id, exam_id, eye)
            result["timings"]["report_ms"] = round((time.perf_counter() - t0_rep) * 1000.0, 1)
            result["timings"]["total_ms"] = round((time.perf_counter() - t0_total) * 1000.0, 1)

            # Backward compatibility pointers
            result["safety_gate_triggered"] = True
            result["screening_status"] = "UNGRADABLE"
            result["message"] = "DOWNSTREAM GRADING HALTED: Image quality is insufficient for diagnostic screening."
            result["action_required"] = quality["recommendation"]
            result["report"] = report_res
            result["grading"] = None
            result["enhancement"] = None
            result["structures"] = None
            result["lesions"] = None
            result["overlays"] = None
            return result

        # ---------------------------------------------------------------------
        # STAGE 2: CLINICAL IMAGE ENHANCEMENT & PREPROCESSING
        # ---------------------------------------------------------------------
        t0_enh = time.perf_counter()
        enhancement = self.enhancement_service.enhance(img_bgr)
        enhanced_bgr = enhancement["enhanced_bgr"]
        t_enh = (time.perf_counter() - t0_enh) * 1000.0
        result["timings"]["enhancement_ms"] = round(t_enh, 1)

        result["preprocessing"]["steps_applied"] = enhancement["steps_applied"]
        result["preprocessing"]["crop_bbox"] = enhancement["crop_bbox"]
        result["preprocessing"]["enhanced_url"] = enhancement["relative_url"]

        # ---------------------------------------------------------------------
        # STAGE 3: RETINAL ANATOMICAL LANDMARKS & VASCULAR DENSITY
        # ---------------------------------------------------------------------
        t0_st = time.perf_counter()
        structures = self.structure_service.detect(enhanced_bgr)
        t_st = (time.perf_counter() - t0_st) * 1000.0
        result["timings"]["anatomy_ms"] = round(t_st, 1)

        od_mask = structures["optic_disc"]["mask_np"]
        vessel_density = structures["vessels"]["density_percent"]

        result["anatomy"]["optic_disc"] = {
            "detected": structures["optic_disc"]["detected"],
            "centroid": structures["optic_disc"]["centroid"],
            "radius_px": structures["optic_disc"]["radius_px"],
            "methodology": structures["optic_disc"]["methodology"]
        }
        result["anatomy"]["fovea"] = structures["fovea"]
        result["anatomy"]["vessels"] = {
            "detected": structures["vessels"]["detected"],
            "density_percent": structures["vessels"]["density_percent"],
            "methodology": structures["vessels"]["methodology"]
        }
        result["anatomy"]["vessel_overlay_url"] = structures["vessel_overlay_url"]
        result["anatomy"]["landmarks_overlay_url"] = structures["structures_overlay_url"]

        # ---------------------------------------------------------------------
        # STAGE 4: CANDIDATE LESION EVIDENCE ANALYSIS & CROPS
        # ---------------------------------------------------------------------
        t0_l = time.perf_counter()
        lesions = self.lesion_service.analyze(
            enhanced_bgr,
            od_mask=od_mask,
            vessel_density=vessel_density
        )
        t_l = (time.perf_counter() - t0_l) * 1000.0
        result["timings"]["lesions_ms"] = round(t_l, 1)
        result["lesion_candidates"] = lesions

        # ---------------------------------------------------------------------
        # STAGE 5: EXP-001 GRADING & GRAD-CAM++ ATTRIBUTION
        # ---------------------------------------------------------------------
        t0_g = time.perf_counter()
        grading = self.grading_service.grade(pil_img, target_class=target_class)
        t_g = (time.perf_counter() - t0_g) * 1000.0
        result["timings"]["grading_ms"] = round(t_g, 1)

        result["model"]["predicted_grade"] = grading["grade"]
        result["model"]["severity_name"] = grading["severity_name"]
        result["model"]["confidence"] = grading["confidence"]
        result["model"]["confidence_percent"] = grading["confidence_percent"]
        result["model"]["confidence_level"] = grading["confidence_level"]
        result["model"]["probabilities"] = grading["probabilities"]
        result["model"]["top_probability"] = grading.get("top_probability", grading["confidence"])
        result["model"]["runner_up_grade"] = grading.get("runner_up_grade")
        result["model"]["runner_up_name"] = grading.get("runner_up_name", "N/A")
        result["model"]["runner_up_probability"] = grading.get("runner_up_probability", 0.0)
        result["model"]["margin"] = grading.get("margin", 0.0)
        result["model"]["margin_percent"] = grading.get("margin_percent", 0.0)
        result["model"]["calibration_status"] = grading.get("calibration_status", "Calibration: not applied (raw model probabilities)")
        result["model"]["referral"] = grading["referral"]
        result["model"]["referral_badge"] = grading["referral_badge"]
        result["model"]["referral_rule"] = grading.get("referral_rule", "Configured prototype screening rule (Grade >= 2)")
        result["model"]["recommendation"] = grading["recommendation"]

        # XAI block
        if "xai" in grading:
            result["xai"] = grading["xai"]
        else:
            result["xai"]["gradcam_url"] = grading.get("gradcam_url")

        # Backward compatibility pointers for existing tests and web client
        result["grading"] = grading
        result["structures"] = result["anatomy"]
        result["lesions"] = lesions
        result["enhancement"] = {
            "enhanced_url": enhancement["relative_url"],
            "steps_applied": enhancement["steps_applied"]
        }

        # ---------------------------------------------------------------------
        # STAGE 6: SCREENING INTERPRETATION & TRIAGE
        # ---------------------------------------------------------------------
        result["screening"]["screening_status"] = "GRADABLE"
        result["screening"]["safety_gate_triggered"] = False
        result["screening"]["referral_urgency"] = "Referral Required" if grading["referral"] else "Routine Follow-up"
        result["screening"]["action_required"] = grading["recommendation"]

        # ---------------------------------------------------------------------
        # STAGE 7: GROUNDED EXPLAINAI ENGINE
        # ---------------------------------------------------------------------
        result["explanation"] = ExplainAIEngine.explain(result)

        # ---------------------------------------------------------------------
        # STAGE 8: CLINICAL SCREENING REPORT GENERATION
        # ---------------------------------------------------------------------
        t0_rep = time.perf_counter()
        report_res = self.report_service.generate(result, patient_id, exam_id, eye)
        t_rep = (time.perf_counter() - t0_rep) * 1000.0
        result["timings"]["report_ms"] = round(t_rep, 1)

        t_total = (time.perf_counter() - t0_total) * 1000.0
        result["timings"]["total_ms"] = round(t_total, 1)

        # Backward compatibility pointers for existing tests and web client
        result["grading"] = grading
        result["structures"] = result["anatomy"]
        result["lesions"] = lesions
        result["enhancement"] = {
            "enhanced_url": enhancement["relative_url"],
            "steps_applied": enhancement["steps_applied"]
        }
        result["safety_gate_triggered"] = False
        result["screening_status"] = "GRADABLE"
        result["original_url"] = original_url
        result["overlays"] = {
            "original": original_url,
            "enhanced": enhancement["relative_url"],
            "vessels": structures["vessel_overlay_url"],
            "lesions": lesions["overlay_url"],
            "heatmap": grading["gradcam_url"]
        }
        result["report"] = report_res

        return result

    def run_region_sensitivity(
        self,
        image_path: Path,
        regions: List[Dict[str, Any]],
        target_class: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes on-demand region sensitivity analysis for up to 3 evidence regions.
        """
        if self.region_analyzer is None:
            return {
                "method": "Model sensitivity to region masking",
                "status": "UNAVAILABLE",
                "message": "Region sensitivity analyzer unavailable (model not initialized).",
                "regions": []
            }

        pil_img = Image.open(image_path).convert("RGB")
        return self.region_analyzer.analyze_regions(
            image_pil=pil_img,
            regions=regions,
            target_class=target_class
        )
