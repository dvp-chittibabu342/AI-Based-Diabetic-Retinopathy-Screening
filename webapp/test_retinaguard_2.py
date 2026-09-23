"""
RetinaGuard 2.0 Comprehensive Verification Test Suite
Tests for:
1. Real Grad-CAM++ attribution (2nd and 3rd order gradients)
2. No synthetic heatmap fallback (honest XAI_UNAVAILABLE refusal)
3. Dynamic target class selection
4. On-demand region sensitivity masking with exact neutral phrasing
5. Grounded deterministic ExplainAI engine (8 questions)
6. Candidate lesion terminology enforcement
7. 5-class probability correctness, decision margin, and calibration label
8. Report and UI AnalysisResult contract parity
9. Ungradable fundus scan safety gate behavior
10. Pipeline transparency registry completeness
11. REST API endpoints verification
"""

import os
import unittest
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image
from fastapi.testclient import TestClient

from webapp.main import app
from python.contracts.analysis_contract import create_empty_analysis_result, CLASS_NAMES, SEVERITY_SCALE
from python.explainability.gradcam_plus_plus import GradCAMPlusPlus, compute_gradcam_plus_plus
from python.explainability.gradcam import GradCAM
from python.explainability.region_sensitivity import RegionSensitivityAnalyzer
from python.explainability.explain_ai import ExplainAIEngine
from python.transparency.pipeline_registry import PipelineRegistry
from webapp.services.grading_service import GradingService
from webapp.services.lesion_service import LesionService
from webapp.services.quality_service import QualityService
from webapp.services.report_service import ReportService
from webapp.services.screening_service import ScreeningService

ROOT = Path(__file__).resolve().parent.parent


class TestRetinaGuard2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.temp_dir = ROOT / "webapp" / "static" / "outputs" / "test_run"
        cls.temp_dir.mkdir(parents=True, exist_ok=True)
        
        cls.grading_service = GradingService(cls.temp_dir)
        cls.lesion_service = LesionService(cls.temp_dir)
        cls.quality_service = QualityService()
        cls.report_service = ReportService(cls.temp_dir)
        cls.screening_service = ScreeningService(cls.temp_dir, cls.temp_dir)
        
        # Load sample demo image (Case 1: Grade 0 or Case 2: Grade 1)
        cls.sample_path = ROOT / "demo" / "sample_images" / "demo_case1_grade0.png"
        if not cls.sample_path.exists():
            cls.sample_path = ROOT / "demo" / "sample_images" / "demo_case2_grade1.png"
        if not cls.sample_path.exists():
            cls.sample_path = ROOT / "demo" / "sample_images" / "demo_grade0.png"
            
        cls.sample_img_bgr = cv2.imread(str(cls.sample_path))
        cls.assertIsNotNone(cls.sample_img_bgr, f"Failed to load sample image from {cls.sample_path}")
        cls.sample_pil = Image.fromarray(cv2.cvtColor(cls.sample_img_bgr, cv2.COLOR_BGR2RGB))
        cls.sample_tensor = cls.grading_service.predictor.preprocessor.preprocess_for_inference(cls.sample_pil)

    # 1. Grad-CAM++ Attribution
    def test_gradcam_plus_plus_computation(self):
        """Verify real Grad-CAM++ produces normalized heatmaps using higher-order gradients."""
        model = self.grading_service.predictor.model
        if model is None:
            self.skipTest("PyTorch model not loaded in environment")
        
        cam_pp = GradCAMPlusPlus(model)
        heatmap, status = cam_pp.generate(self.sample_tensor, target_class=0)
        cam_pp.remove_hooks()
        
        self.assertEqual(status, "SUCCESS")
        self.assertIsNotNone(heatmap, "Grad-CAM++ must return a valid heatmap")
        self.assertEqual(heatmap.shape, (self.sample_tensor.shape[2], self.sample_tensor.shape[3]))
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0)
        # Saliency should not be a flat solid constant
        self.assertGreater(float(heatmap.max() - heatmap.min()), 0.01)

    # 2. No Synthetic Heatmap Fallback
    def test_no_synthetic_heatmap_fallback(self):
        """Verify that when the model is None or errors, no synthetic Gaussian blobs are fabricated."""
        # Test GradCAM with None model
        cam = GradCAM(None)
        heatmap, overlay = cam.generate(self.sample_tensor, target_class=0)
        self.assertIsNone(overlay, "Overlay must be None when model is unavailable")
        self.assertEqual(float(heatmap.max()), 0.0, "Heatmap must be empty array, never synthetic Gaussian blobs")
        
        # Test GradCAMPlusPlus with None model
        heatmap_pp = compute_gradcam_plus_plus(None, self.sample_tensor, target_class=0)
        self.assertIsNone(heatmap_pp, "Grad-CAM++ must return None on unavailable model")

    # 3. Dynamic Target Class Selection
    def test_dynamic_target_class_selection(self):
        """Verify Grad-CAM++ computes distinct heatmaps for different target classes."""
        model = self.grading_service.predictor.model
        if model is None:
            self.skipTest("PyTorch model not loaded")
        
        cam_pp = GradCAMPlusPlus(model)
        heatmap_c0, _ = cam_pp.generate(self.sample_tensor, target_class=0)
        heatmap_c4, _ = cam_pp.generate(self.sample_tensor, target_class=4)
        cam_pp.remove_hooks()
        
        self.assertIsNotNone(heatmap_c0)
        self.assertIsNotNone(heatmap_c4)
        # Class 0 and Class 4 gradients should produce differing attribution maps
        diff = np.abs(heatmap_c0.astype(np.float32) - heatmap_c4.astype(np.float32)).mean()
        self.assertGreater(float(diff), 1e-5, "Attribution should be class-specific")

    # 4. Region Sensitivity Analysis
    def test_region_sensitivity_masking(self):
        """Verify on-demand region sensitivity masking preserves weights and produces neutral phrasing."""
        model = self.grading_service.predictor.model
        if model is None:
            self.skipTest("PyTorch model not loaded")
            
        analyzer = RegionSensitivityAnalyzer(model, self.grading_service.predictor.preprocessor)
        regions = [
            {"region_id": "cand_1", "region_name": "Perimacular Microaneurysm Candidate", "region_type": "Microaneurysm", "bbox": [100, 100, 60, 60]},
            {"region_id": "cand_2", "region_name": "Inferior Hard Exudate Candidate", "region_type": "Hard Exudate", "bbox": [200, 200, 50, 50]}
        ]
        
        result = analyzer.analyze(self.sample_img_bgr, candidate_regions=regions, target_class=2)
        self.assertIn("results", result)
        self.assertEqual(len(result["results"]), 2)
        
        for r in result["results"]:
            self.assertIn("original_prob", r)
            self.assertIn("masked_prob", r)
            self.assertIn("prob_delta", r)
            self.assertIn("delta_percent", r)
            self.assertIn("statement", r)
            # Neutral phrasing requirement
            self.assertIn("Model sensitivity to region masking", r["statement"])
            self.assertNotIn("causal proof", r["statement"].lower())
            self.assertNotIn("proved", r["statement"].lower())

    # 5. ExplainAI Deterministic Grounding
    def test_explainai_deterministic_grounding(self):
        """Verify ExplainAI answers all 8 mandatory questions strictly using AnalysisResult fields."""
        result = self.screening_service.run_screening(
            image_path=self.sample_path,
            original_url="/demo/sample_images/demo_test.png"
        )
        self.assertIn("explanation", result)
        
        explanation = ExplainAIEngine.generate_explanation(result)
        self.assertEqual(len(explanation["questions"]), 8, "ExplainAI must answer exactly 8 mandatory screening questions")
        
        expected_keywords = [
            ["accepted", "rejected"],
            ["preprocessing", "filter"],
            ["grade", "predicted"],
            ["influence", "region"],
            ["candidate", "finding"],
            ["probability", "distribution"],
            ["limitation"],
            ["screener", "understand"]
        ]
        
        for i, keywords in enumerate(expected_keywords):
            q_obj = explanation["questions"][i]
            q_text = q_obj["question"].lower()
            self.assertTrue(any(kw in q_text for kw in keywords), f"Question {i+1} ({q_text}) should match keywords {keywords}")
            self.assertTrue(len(q_obj["answer"]) > 20, f"Answer for question {i+1} must be thorough and grounded")
            self.assertIsInstance(q_obj["grounding_fields"], list)
            self.assertGreater(len(q_obj["grounding_fields"]), 0)

    # 6. Candidate Lesion Terminology
    def test_candidate_lesion_evidence_framing(self):
        """Ensure all lesion findings are framed strictly as candidate evidence, never confirmed lesions."""
        lesion_data = self.lesion_service.detect(self.sample_img_bgr)
        
        # Microaneurysms, Exudates, Hemorrhages must all have candidate_count
        self.assertIn("candidate_count", lesion_data["microaneurysms"])
        self.assertIn("candidate_count", lesion_data["hard_exudates"])
        self.assertIn("candidate_count", lesion_data["hemorrhages"])
        
        # Neovascularization must be labeled as heuristic
        self.assertIn("density_heuristic", lesion_data["neovascularization"])
        self.assertNotIn("confirmed", lesion_data["neovascularization"]["density_heuristic"].lower())
        
        # Candidate visual crops
        crops = lesion_data.get("visual_crops", [])
        for c in crops:
            self.assertIn("candidate_type", c)
            self.assertIn("candidate", c["candidate_type"].lower())

    # 7. Probability Correctness and Calibration Label
    def test_probability_distribution_and_decision_margin(self):
        """Verify 5-class distribution sums to 1.0, top margin is calculated, and uncalibrated label is present."""
        result = self.screening_service.run_screening(
            image_path=self.sample_path,
            original_url="/demo/sample_images/demo_test.png"
        )
        g = result["grading"]
        
        self.assertEqual(len(g["probabilities"]), 5)
        total_p = sum(p["probability"] for p in g["probabilities"])
        self.assertAlmostEqual(total_p, 1.0, places=2)
        
        # Calibration label
        self.assertIn("Calibration: not applied", g["calibration_status"])
        
        # Decision margin
        self.assertIsNotNone(g["decision_margin_percent"])
        self.assertGreaterEqual(g["decision_margin_percent"], 0.0)
        self.assertIsNotNone(g["runner_up_grade"])

    # 8. Report and UI Parity
    def test_report_and_ui_parity(self):
        """Verify the generated HTML report contains the exact fields and values from AnalysisResult."""
        result = self.screening_service.run_screening(
            image_path=self.sample_path,
            original_url="/demo/sample_images/demo_test.png"
        )
        report_meta = self.report_service.generate_html_report(result)
        report_path = report_meta["filepath"]
        self.assertTrue(os.path.exists(report_path))
        
        with open(report_path, "r", encoding="utf-8") as f:
            html = f.read()
            
        # Check parity of key values
        self.assertIn(f"Grade {result['grading']['grade']}", html)
        self.assertIn(result["grading"]["severity_name"], html)
        self.assertIn(f"{result['grading']['confidence_percent']}%", html)
        self.assertIn("Candidate Evidence", html)
        self.assertIn("IMPORTANT CLINICAL DISCLAIMER", html)

    # 9. Ungradable Safety Behavior (Case 6)
    def test_ungradable_safety_behavior(self):
        """Verify an ungradable fundus image triggers the safety gate, halts grading, and explains the reason."""
        ungradable_path = ROOT / "demo" / "sample_images" / "demo_case6_ungradable.png"
        if not ungradable_path.exists():
            ungradable_path = ROOT / "demo" / "sample_images" / "demo_ungradable.png"
            
        self.assertTrue(ungradable_path.exists(), "Ungradable demo image must exist")
        
        result = self.screening_service.run_screening(
            image_path=ungradable_path,
            original_url="/demo/sample_images/demo_case6_ungradable.png"
        )
        self.assertTrue(result["safety_gate_triggered"], "Safety gate must trigger on ungradable image")
        self.assertFalse(result["quality"]["gradable"], "Image must be flagged ungradable")
        self.assertIsNone(result["grading"], "Grading must be halted (result['grading'] is None)")
        self.assertEqual(result["model"]["predicted_grade"], -1, "Grading must be halted (grade -1)")
        self.assertEqual(result["model"]["severity_name"], "UNGRADABLE")
        self.assertEqual(len(result["model"]["probabilities"]), 0, "Probabilities must not be fabricated on ungradable scans")
        self.assertIn("recapture", result["quality"]["recommendation"].lower())

    # 10. Pipeline Transparency Registry
    def test_pipeline_transparency_registry(self):
        """Verify all 12 pipeline stages are registered with full clinical audit information."""
        stages = PipelineRegistry.get_all_stages()
        self.assertEqual(len(stages), 12, "Pipeline must register all 12 stages")
        
        required_keys = ["stage_id", "name", "category", "what_it_does", "why_used", "actual_parameters", "what_changed_in_image", "limitation"]
        for s in stages:
            for k in required_keys:
                self.assertIn(k, s, f"Stage {s.get('stage_id')} missing key {k}")
                self.assertTrue(len(str(s[k])) > 0, f"Stage {s.get('stage_id')} key {k} cannot be empty")

    # 11. API Endpoints for Stages and Sensitivity
    def test_api_endpoints_retinaguard_2(self):
        """Verify /api/pipeline-stages and /api/region-sensitivity endpoints work correctly."""
        # 1. Pipeline stages list
        resp = self.client.get("/api/pipeline-stages")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data), 12)
        
        # 2. Single pipeline stage
        resp_stage = self.client.get("/api/pipeline-stages/stage_1_quality_gate")
        self.assertEqual(resp_stage.status_code, 200)
        self.assertEqual(resp_stage.json()["stage_id"], "stage_1_quality_gate")
        
        # 3. 6 Demo cases including ungradable
        resp_demo = self.client.get("/api/demo-cases?all=true")
        self.assertEqual(resp_demo.status_code, 200)
        demo_list = resp_demo.json()
        self.assertEqual(len(demo_list), 6)
        self.assertTrue(any(c["id"] == "case_6" for c in demo_list))


if __name__ == "__main__":
    unittest.main()
