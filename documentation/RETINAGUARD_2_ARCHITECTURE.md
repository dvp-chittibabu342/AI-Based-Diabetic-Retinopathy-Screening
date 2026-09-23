# RetinaGuard 2.0 — System Architecture Document
**SIH 2026 Problem Statement: SIH26038**  
**Team: VyuhaX**  
**Backbone: EXP-001 (EfficientNet-B0)**  
**Runtime: Offline Edge CPU / Local Workstation**  

---

## 1. Executive Architectural Overview

RetinaGuard 2.0 is an offline-first, explainable, evidence-grounded AI screening workstation for Diabetic Retinopathy (DR). Engineered specifically for low-resource primary healthcare centers (PHCs) and mobile tele-ophthalmology screening vans, RetinaGuard 2.0 transforms raw digital fundus camera images into actionable, transparent clinical decision support within 400 milliseconds on standard CPU hardware.

Unlike traditional "black-box" deep learning classifiers that merely output a single scalar probability, RetinaGuard 2.0 implements a **12-stage transparent processing pipeline** where:
1. Every preprocessing and filtering step is audited and explainable.
2. Microvascular lesions are detected as **candidate evidence** rather than autonomous diagnoses.
3. Neural decisions are localized using **Grad-CAM++** attribution maps with zero synthetic fallbacks.
4. Screener questions are answered deterministically by a grounded **ExplainAI** engine.
5. All UI panels, API endpoints, and exported PDF/HTML clinical reports derive from a single unified contract: `AnalysisResult`.

```
Raw Fundus Photograph (JPG / PNG / TIFF)
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ STAGE 1: Image Quality Assessment Gate                  │
│ • Laplacian Variance (Sharpness)                        │
│ • Green-Channel Mean (Illumination)                     │
│ • Retinal Pixel StdDev (Contrast)                       │
│ • Active Sensor Coverage (Field of View)                │
└────────────┬───────────────────────────────┬────────────┘
             │ [Optical Score < 40.0]        │ [Optical Score >= 40.0]
             ▼                               ▼
┌─────────────────────────┐     ┌────────────────────────────────────────────────────────┐
│ SAFETY GATE TRIGGERED   │     │ STAGES 2–4: Optical Standardisation & Filtering        │
│ • Downstream Halted     │     │ • Border/Margin Removal                                │
│ • Grade = -1 UNGRADABLE │     │ • Canonical Scale Standardisation (512x512)            │
│ • Actionable Recapture  │     │ • Green-Channel CLAHE (clipLimit=2.0, tileGrid=8x8)    │
│   Guidance Generated    │     └───────────────────────────┬────────────────────────────┘
└────────────┬────────────┘                                 │
             │                                              ▼
             │                  ┌────────────────────────────────────────────────────────┐
             │                  │ STAGES 5–6: Retinal Anatomy & Candidate Lesions        │
             │                  │ • Optic Disc Peak Luminance & Morphology               │
             │                  │ • Fovea Centralis Geometric Projection                 │
             │                  │ • Multi-scale Vessel Segmentation & Density %          │
             │                  │ • Candidate Extraction (MAs, EX, HE, SE) + 96x96 Crops │
             │                  │ • Neovascularization Risk Density Heuristic Indicator  │
             │                  └───────────────────────────┬────────────────────────────┘
             │                                              │
             │                                              ▼
             │                  ┌────────────────────────────────────────────────────────┐
             │                  │ STAGES 7–8: Deep Learning Grading & Attribution        │
             │                  │ • EfficientNet-B0 (EXP-001) ONNX Runtime Engine        │
             │                  │ • 5-Class Softmax Vector (Grades 0 to 4)               │
             │                  │ • Top Probability, Runner-Up, and Margin %             │
             │                  │ • Primary: Grad-CAM++ (2nd/3rd order gradients)        │
             │                  │ • Fallback: Grad-CAM (Zero synthetic fallbacks)        │
             │                  └───────────────────────────┬────────────────────────────┘
             │                                              │
             │                                              ▼
             │                  ┌────────────────────────────────────────────────────────┐
             │                  │ STAGES 9–10: Perturbation & Grounded Reasoning         │
             │                  │ • On-Demand Candidate Region Masking Sensitivity       │
             │                  │ • Deterministic ExplainAI Engine (8 Screening Answers) │
             │                  │ • Prototype Screening Triage (Grade >= 2 Referral)     │
             │                  └───────────────────────────┬────────────────────────────┘
             │                                              │
             └──────────────────────┬───────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ UNIFIED CONTRACT: AnalysisResult JSON Schema                                           │
│ Sourced synchronously into:                                                            │
│ 1. Clinical Screening Workstation UI (HTML5 / Vanilla CSS / ES6)                       │
│ 2. REST API Responses (/api/analyze, /api/region-sensitivity)                          │
│ 3. Self-Contained Printable Clinical HTML Report                                       │
│ 4. Verification Test Suites                                                            │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure & File Inventory

The RetinaGuard repository maintains strict modular separation between backend services, deep learning inference, explainability engines, web presentation, and MATLAB legacy validation components:

```
d:\RetinaGuard-main\RetinaGuard-main\
├── app/                                 # MATLAB Graphical Application Stack
│   ├── RetinaGuardApp.m                 # MATLAB standalone desktop screening workstation
│   └── RetinaGuardApp.mlapp             # MATLAB App Designer binary packaging
├── core/                                # MATLAB Classical Computer Vision Modules
│   ├── qualityGate.m                    # Optical quality screening gate
│   ├── preprocessFundus.m               # Circular aperture extraction & CLAHE
│   ├── detectOpticDisc.m                # Luminance-based optic disc localization
│   ├── detectFovea.m                    # Geometric fovea projection
│   ├── segmentVessels.m                 # Multi-scale 2D Gabor vessel filter
│   ├── detectLesions.m                  # Heuristic candidate lesion extractor
│   ├── classifyDR.m                     # ICDR staging decision engine
│   └── generateReport.m                 # Medical ASCII/HTML report compiler
├── demo/                                # Curated Authentic Reference Datasets
│   ├── sample_images/
│   │   ├── demo_case1_grade0.png        # Case 1: Normal Retina (Grade 0)
│   │   ├── demo_case2_grade1.png        # Case 2: Mild NPDR (Grade 1)
│   │   ├── demo_case3_grade2.png        # Case 3: Moderate NPDR (Grade 2)
│   │   ├── demo_case4_grade3.png        # Case 4: Severe NPDR (Grade 3)
│   │   ├── demo_case5_grade4.png        # Case 5: Proliferative DR (Grade 4)
│   │   └── demo_case6_ungradable.png    # Case 6: Poor Quality / Recapture Required
│   ├── cases.json                       # Clinical case metadata registry
│   └── sample_cases.json                # Web client demo configuration
├── documentation/                       # Clinical, Model, & Architectural Specifications
│   ├── RETINAGUARD_2_AUDIT.md           # Baseline technical audit
│   ├── RETINAGUARD_2_ARCHITECTURE.md    # This architectural specification
│   ├── RETINAGUARD_2_XAI.md             # Attribution & explainability whitepaper
│   ├── RETINAGUARD_2_MODEL_CARD.md      # EXP-001 empirical validation & training details
│   ├── RETINAGUARD_2_LIMITATIONS.md     # Clinical boundaries & ethical safety guidelines
│   └── RETINAGUARD_2_DEMO_SCRIPT.md     # Hackathon jury demonstration protocol
├── models/                              # Locked Production Model Weights
│   ├── aptos_efficientnet/
│   │   ├── best_model.onnx              # Production EXP-001 ONNX runtime graph
│   │   ├── best_model.pt                # Production PyTorch weights (for Grad-CAM++)
│   │   ├── class_mapping.json           # Locked 5-class ICDR severity mapping
│   │   └── training_history.json        # Training convergence epoch metrics
│   └── experiments/
│       └── lesion_segmentation/
│           └── lesion_unet.onnx         # IDRiD-trained dual-head candidate lesion U-Net
├── python/                              # Python Screening Engine & Model Core
│   ├── config/config.yaml               # Global system configuration
│   ├── contracts/
│   │   └── analysis_contract.py         # Unified AnalysisResult contract schema
│   ├── explainability/
│   │   ├── gradcam.py                   # Standard Grad-CAM with zero synthetic blobs
│   │   ├── gradcam_plus_plus.py         # Grad-CAM++ with higher-order gradients
│   │   ├── region_sensitivity.py        # On-demand candidate region masking analyzer
│   │   └── explain_ai.py                # Deterministic grounded explanation engine
│   ├── inference/
│   │   ├── predict.py                   # Production DRPredictor runtime orchestrator
│   │   └── segment_lesions.py           # Lesion U-Net inference & candidate bounding boxes
│   ├── preprocessing/
│   │   └── retinal_preprocessor.py      # Standardized training & inference transformations
│   ├── transparency/
│   │   └── pipeline_registry.py         # 12-stage transparency registry
│   └── verify_pipeline.py               # Core pipeline validation script
├── webapp/                              # FastAPI Clinical Screening Workstation
│   ├── main.py                          # Application entry point & REST endpoints
│   ├── run.py                           # CLI bootstrap launcher
│   ├── services/
│   │   ├── grading_service.py           # EXP-001 grading & Grad-CAM++ service
│   │   ├── quality_service.py           # Optical safety gate service
│   │   ├── structure_service.py         # Anatomy & vessel density service
│   │   ├── lesion_service.py            # Candidate lesion & visual crops service
│   │   ├── enhancement_service.py       # Border crop & green-channel CLAHE service
│   │   ├── report_service.py            # Single-source HTML report generator
│   │   └── screening_service.py         # Master pipeline orchestrator
│   ├── static/
│   │   ├── css/style.css                # Clinical workstation styles
│   │   └── js/app.js                    # Dynamic workstation controller
│   ├── templates/
│   │   └── index.html                   # 8-stage interactive UI template
│   ├── test_webapp.py                   # Baseline webapp test suite (16 tests)
│   └── test_retinaguard_2.py            # RetinaGuard 2.0 verification suite (11 tests)
├── launch.m                             # MATLAB desktop workstation launcher
├── runAllTests.m                        # MATLAB test runner
└── README.md                            # Primary project documentation
```

---

## 3. The Central Single Source of Truth: `AnalysisResult` Contract

To prevent divergence between what is displayed on the screen, what is computed by ExplainAI, and what is recorded in clinical referral reports, RetinaGuard 2.0 enforces a strict single-source-of-truth architecture implemented in `python/contracts/analysis_contract.py`.

Every analysis generates one immutable dictionary containing these primary top-level namespaces:

```json
{
  "contract_version": "2.0.0",
  "metadata": {
    "analysis_id": "RG2-20260923101500-A9F4B1",
    "timestamp": "2026-09-23T10:15:00.124512",
    "model_version": "EXP-001",
    "architecture": "EfficientNet-B0 (Focal Loss, Balanced Sampling)",
    "device": "CPU (Offline Edge Runtime)",
    "image_hash": "e3b0c44298fc1c14"
  },
  "input": {
    "patient_id": "PT-82910",
    "exam_id": "EX-00412",
    "eye": "Right (OD)",
    "width": 2144,
    "height": 1424,
    "original_url": "/static/uploads/pt82910_right.png",
    "demo_reference": null
  },
  "quality": {
    "score": 84.5,
    "status": "GOOD",
    "gradable": true,
    "focus": "Good",
    "illumination": "Good",
    "contrast": "Good",
    "field_of_view": "Good",
    "laplacian_var": 92.4,
    "mean_lum": 118.2,
    "fov_percent": 91.5,
    "reason": "Image meets optical clarity criteria for diagnostic grading.",
    "recommendation": "Proceed with screening.",
    "threshold_notes": "Configured prototype thresholds: minimum composite score 40.0, sharpness variance >= 35.0, illumination in [30.0, 225.0]."
  },
  "preprocessing": {
    "steps_applied": ["Border/margin crop", "Green-channel extraction", "Contrast Limited Adaptive Histogram Equalization (CLAHE)"],
    "crop_bbox": [82, 0, 1980, 1424],
    "enhanced_url": "/static/outputs/enhanced_ex00412.png",
    "resolution_standardized": [512, 512]
  },
  "anatomy": {
    "optic_disc": {
      "detected": true,
      "centroid": [186, 109],
      "radius_px": 28,
      "methodology": "Computer Vision Peak Luminance & Morphology"
    },
    "fovea": {
      "detected": true,
      "coordinates": [144, 109],
      "radius_px": 14,
      "methodology": "Anatomical Temporal Geometric Projection"
    },
    "vessels": {
      "detected": true,
      "density_percent": 11.11,
      "methodology": "Multi-scale Morphological Black Top-Hat & Ridge Filtering"
    }
  },
  "lesion_candidates": {
    "mode": "CANDIDATE DETECTION",
    "badge_label": "CANDIDATE EVIDENCE",
    "is_ai": true,
    "disclaimer": "Candidate evidence regions identified by computer vision; not independently confirmed clinical lesions.",
    "microaneurysms": { "candidate_count": 15, "total_area_px": 38, "label": "Microaneurysm candidates" },
    "hard_exudates": { "candidate_count": 0, "total_area_px": 0, "label": "Hard exudate candidates" },
    "hemorrhages": { "candidate_count": 1, "total_area_px": 84, "label": "Hemorrhage candidates" },
    "soft_exudates": { "candidate_count": 0, "total_area_px": 0, "label": "Soft exudate candidates" },
    "neovascularization": {
      "risk_level": "Low",
      "label": "Neovascularization Risk Indicator",
      "density_heuristic": "Vessel density stratification heuristic indicator"
    },
    "visual_crops": [
      {
        "candidate_type": "Microaneurysm Candidate",
        "bounding_box": { "x": 142, "y": 88, "width": 96, "height": 96 },
        "crop_base64": "/9j/4AAQSkZJRg...",
        "description": "Focal microvascular dilatation candidate."
      }
    ]
  },
  "model": {
    "predicted_grade": 2,
    "severity_name": "Moderate DR",
    "confidence": 0.400,
    "confidence_percent": 40.0,
    "confidence_level": "Moderate",
    "probabilities": [
      { "grade": 0, "label": "No DR", "probability": 0.0512, "percentage": 5.1 },
      { "grade": 1, "label": "Mild DR", "probability": 0.2314, "percentage": 23.1 },
      { "grade": 2, "label": "Moderate DR", "probability": 0.4002, "percentage": 40.0 },
      { "grade": 3, "label": "Severe DR", "probability": 0.2811, "percentage": 28.1 },
      { "grade": 4, "label": "Proliferative DR", "probability": 0.0361, "percentage": 3.6 }
    ],
    "top_probability": 0.4002,
    "runner_up_grade": 3,
    "runner_up_name": "Severe DR",
    "runner_up_probability": 0.2811,
    "margin": 0.1191,
    "margin_percent": 11.9,
    "calibration_status": "Calibration: not applied (raw model probabilities)",
    "referral": true,
    "referral_badge": "REFERABLE DR",
    "referral_rule": "Configured prototype screening rule (Grade >= 2)",
    "recommendation": "Moderate non-proliferative DR. Ophthalmology evaluation recommended within 3–6 months."
  },
  "xai": {
    "method": "Grad-CAM++",
    "status": "SUCCESS",
    "target_class": 2,
    "target_class_name": "Moderate DR",
    "target_layer": "backbone.features.8.0 (final Conv2d, 1280 channels)",
    "gradcam_url": "/static/outputs/gradcam_ex00412.png",
    "notice": "Attribution visualization shows regions contributing to model prediction; it is not confirmed lesion segmentation."
  },
  "region_sensitivity": {
    "method": "Model sensitivity to region masking",
    "target_class": 2,
    "original_probability": 0.4002,
    "regions_tested_count": 0,
    "regions": [],
    "disclaimer": "Model sensitivity to region masking measures local perturbation response. It is not causal proof or manual lesion validation."
  },
  "explanation": {
    "engine": "ExplainAI (Deterministic, Evidence-Grounded)",
    "questions": [
      {
        "id": "q1_acceptance",
        "question": "Why was this image accepted?",
        "answer": "The retinal photograph met all optical quality gate criteria...",
        "grounding_fields": ["quality.score", "quality.status", "quality.laplacian_var", "quality.fov_percent"]
      }
    ]
  },
  "screening": {
    "screening_status": "GRADABLE",
    "safety_gate_triggered": false,
    "referral_urgency": "Referral Required",
    "action_required": "Ophthalmology evaluation recommended within 3–6 months.",
    "disclaimer": "AI-assisted screening prototype. Not an autonomous clinical diagnosis. Confirm all findings with a licensed ophthalmologist."
  }
}
```

---

## 4. Pipeline Execution Workflow

The end-to-end execution flow of `ScreeningService.run_screening()` proceeds through 8 distinct logical steps:

1. **Input Validation & Hashing**:
   - The raw image bytes are hashed via SHA-256 (truncated to 16 hex characters) for audit trail tracking.
   - The dimensions ($H \times W$) and patient metadata are mapped into the contract.

2. **Optical Quality Assessment**:
   - Evaluated via `QualityService.assess()`.
   - Computes Laplacian variance on the green channel to quantify optical sharpness.
   - Computes sensor coverage to assess field of view (FOV).
   - If composite score $< 40.0$ or sharpness variance $< 35.0$, `safety_gate_triggered` is set to `True`. Downstream inference is immediately bypassed, and the response is returned with actionable recapture instructions.

3. **Clinical Image Enhancement**:
   - Automatically removes unexposed sensor margins (black camera borders).
   - Resamples fundus to a standardized $512 \times 512$ canvas.
   - Applies green-channel Contrast Limited Adaptive Histogram Equalization (CLAHE) with `clipLimit=2.0` and `tileGridSize=(8, 8)` to maximize vessel and lesion contrast while preserving dynamic range.

4. **Anatomical Landmark Localization**:
   - Locates Optic Disc (OD) centroid and radius using green-channel luminance peaking and circular morphological closures.
   - Projects Fovea Centralis coordinates using canonical anatomical temporal offsets relative to the disc center.
   - Extracts retinal vascular tree using multi-scale ridge and black top-hat filtering to calculate vascular area density percentage.

5. **Candidate Lesion Evidence Detection**:
   - Executes IDRiD dual-head U-Net ONNX model (`lesion_unet.onnx`) with classical computer vision fallback.
   - Quantifies microaneurysm candidates, hard exudate candidates, hemorrhage candidates, and soft exudate candidates.
   - Crops up to 6 prominent candidate findings ($96 \times 96$ pixels) for screener verification.
   - Calculates Neovascularization Risk Indicator based on vessel density thresholds.

6. **EXP-001 DR Severity Grading**:
   - Executes preprocessed $224 \times 224$ normalized tensor through locked EfficientNet-B0 ONNX session.
   - Obtains 5-class uncalibrated model probabilities via softmax.
   - Extracts top predicted class, runner-up class, and decision margin percentage.
   - Evaluates prototype screening rule: $\text{Grade} \ge 2 \implies \text{Referable DR}$.

7. **Real Grad-CAM++ Attribution**:
   - Computes second- and third-order analytical gradients of the target class score with respect to `features.8.0` (final Conv2d block, 1280 channels).
   - Blends jet-colormap heatmap with alpha $0.45$ over the retinal background.
   - Enforces zero synthetic blobs: returns explicit `XAI_UNAVAILABLE` on model failure.

8. **Deterministic ExplainAI Generation & Report Compilation**:
   - Passes complete `AnalysisResult` into `ExplainAIEngine.explain()`, answering all 8 screening questions using verified numbers.
   - Compiles full self-contained HTML clinical screening report in `webapp/static/outputs/`.

---

## 5. Offline Edge Constraints & Deployment

| Constraint | Architectural Strategy |
| :--- | :--- |
| **Zero Internet Requirement** | All models (EfficientNet-B0, Lesion U-Net) and libraries run entirely on local CPU runtime. No external API calls. |
| **Low-End Hardware Target** | Sub-400ms end-to-end latency achieved on standard quad-core Intel/AMD x86_64 CPUs without requiring a discrete GPU. |
| **Deterministic Consistency** | Fixed random seeds, constant preprocessor normalization ($\mu=[0.485, 0.456, 0.406], \sigma=[0.229, 0.224, 0.225]$), and immutable ONNX inference graphs. |
| **Audit Compliance** | Every screening session outputs a self-contained HTML report containing SHA-256 image hash, complete 5-class probability vector, candidate counts, and timestamp. |
