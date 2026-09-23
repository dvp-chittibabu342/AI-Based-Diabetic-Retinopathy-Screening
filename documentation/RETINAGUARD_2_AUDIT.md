# RetinaGuard 2.0 — Phase 0 Codebase & Architecture Audit

**Date:** 2026-09-23  
**Project:** RetinaGuard — AI-Assisted Diabetic Retinopathy Screening System  
**Competition / Problem Statement:** SIH 2026 PS: SIH26038  
**Team:** VyuhaX  
**Auditor:** Antigravity AI  

---

## 1. Executive Summary

This comprehensive audit was executed prior to any source code modifications as mandated by **Phase 0**. The RetinaGuard repository was thoroughly analyzed across both its Python (FastAPI/PyTorch/ONNX) and MATLAB stacks. All existing tests were executed in their baseline environment, recording exact pass and failure behaviors.

The repository represents a functional, high-performing research prototype with a validated EfficientNet-B0 backbone (EXP-001) trained on APTOS 2019, an ONNX lesion segmentation U-Net, an image quality assessment gate, classical and DL anatomical/lesion feature extraction, and an HTML clinical report generator.

The goal of the **RetinaGuard 2.0** evolution is to transform this prototype into a transparent, evidence-centered, explainable screening workstation adhering strictly to:
- A single shared `AnalysisResult` contract
- Real, validated Grad-CAM++ with Grad-CAM fallback (zero synthetic heatmaps)
- Explicit candidate lesion framing (microaneurysm, exudate, hemorrhage candidates)
- On-demand region sensitivity masking
- Grounded, offline ExplainAI explaining structured facts
- Transparent model probability vectors and calibrated terminology
- Rejection of ungradable scans with explicit clinical recapture guidance

---

## 2. Complete Repository Structure & Module Identification

```
d:/RetinaGuard-main/RetinaGuard-main/
├── .gitignore
├── .python-version                   # Python 3.11.9 pinned
├── DEPLOYMENT.md
├── README.md
├── requirements.txt
├── launch.m                          # MATLAB GUI launcher
├── runAllTests.m                     # MATLAB automated test suite
├── app/
│   └── RetinaGuardApp.m              # MATLAB App Designer clinical application
├── config/
│   └── RGConfig.m                    # MATLAB global configuration
├── core/                             # MATLAB Core Screening Pipeline
│   ├── enhancement/                  # enhanceRetinalImage.m, removeBlackBorder.m
│   ├── grading/                      # gradeDR.m, runONNXModel.m, runPythonInference.m
│   ├── lesions/                      # detectLesions.m, runLesionSegmentation.m
│   ├── quality/                      # assessImageQuality.m
│   ├── reporting/                    # generateReport.m
│   ├── screening/                    # runScreeningPipeline.m
│   └── structures/                   # detectStructures.m
├── demo/
│   ├── generateSyntheticFundus.m
│   ├── getDemoCases.m                # 6 defined cases (Grades 0-4 + Ungradable)
│   └── sample_images/
│       ├── demo_grade0.png           # Grade 0: Normal retina
│       ├── demo_grade1.png           # Grade 1: Mild NPDR
│       ├── demo_grade2.png           # Grade 2: Moderate NPDR
│       ├── demo_grade3.png           # Grade 3: Severe NPDR
│       └── demo_grade4.png           # Grade 4: Proliferative DR
├── documentation/                    # Technical & clinical documentation
│   ├── AI_PIPELINE.md
│   ├── ARCHITECTURE.md
│   ├── DEPLOYMENT.md
│   ├── EVALUATION.md
│   └── ...
├── models/
│   ├── aptos_efficientnet/           # Production Model EXP-001
│   │   ├── best_model.onnx           # Primary fast inference engine (710 KB graph)
│   │   ├── best_model.onnx.data      # ONNX weights (16 MB)
│   │   ├── best_model.pt             # PyTorch weights (16.3 MB)
│   │   ├── class_mapping.json        # 5-class ICDR labels
│   │   ├── config.yaml
│   │   ├── metadata.json             # QWK=0.7342, Acc=69.85%
│   │   └── metrics.json
│   ├── baseline_model/               # Baseline reference
│   ├── dr/
│   │   └── DRClassifierModel.m       # MATLAB model class
│   └── experiments/
│       ├── EXP-001/                  # Production experiment artifacts
│       └── lesion_segmentation/
│           ├── config.yaml
│           ├── lesion_unet.onnx      # Dual-Head U-Net ONNX (13.4 MB)
│           ├── metrics.json
│           └── training_log.json
├── python/
│   ├── config/config.yaml            # Master Python config
│   ├── explainability/
│   │   └── gradcam.py                # Grad-CAM hook implementation
│   ├── inference/
│   │   ├── predict.py                # DRPredictor class & CLI
│   │   └── segment_lesions.py        # LesionSegmentationPipeline (ONNX + Heuristic)
│   ├── models/
│   │   └── efficientnet_dr.py        # DRClassifier PyTorch architecture
│   ├── preprocessing/
│   │   └── retinal_preprocessor.py   # Border crop, CLAHE, ImageNet normalization
│   ├── segmentation/                 # DualHeadUNet PyTorch architecture & training
│   ├── test_end_to_end_sih.py        # 18 end-to-end integration tests
│   ├── verify_pipeline.py            # Diagnostic script
│   └── verify_walkthrough_cases.py   # Empirical test on 6 cases + 7 error modes
├── webapp/
│   ├── __init__.py                   # Starlette/FastAPI router compatibility shim
│   ├── main.py                       # FastAPI application & REST API routes
│   ├── run.py                        # Uvicorn launcher
│   ├── services/
│   │   ├── enhancement_service.py    # CLAHE, border crop, 512x512 standardization
│   │   ├── grading_service.py        # EXP-001 ONNX/PyTorch inference & Grad-CAM
│   │   ├── lesion_service.py         # AI segmentation & heuristic lesion candidate detection
│   │   ├── quality_service.py        # Laplacian sharpness, luminance, contrast, FOV
│   │   ├── report_service.py         # Self-contained HTML report generator
│   │   ├── screening_service.py      # Master pipeline orchestrator
│   │   └── structure_service.py      # Optic disc, fovea, vessel density detection
│   ├── static/
│   │   ├── css/style.css             # Workstation styling
│   │   └── js/app.js                 # Workstation frontend logic
│   ├── templates/
│   │   └── index.html                # Workstation SPA interface
│   ├── test_live_server.py           # Integration script testing live port 8000
│   └── test_webapp.py                # 16-test comprehensive webapp test suite
└── tests/
    └── test_lesion_segmentation.py   # 9 unit tests for lesion segmentation
```

---

## 3. Subsystem Breakdown & Implementation Identification

### 3.1 FastAPI Application Entry Point & Routes
- **Entry point:** `webapp/main.py` (instantiates `FastAPI`, mounts `/static`, `/uploads`, `/outputs`, `/demo/sample_images`).
- **Server runner:** `webapp/run.py` (executes `uvicorn.run("webapp.main:app", host="127.0.0.1", port=8000)`).
- **Current Routes:**
  1. `GET /` — Serves `webapp/templates/index.html`.
  2. `GET /health` — Returns JSON with model name (`EXP-001`), architecture (`EfficientNet-B0`), offline status (`true`), and held-out validation metrics (QWK: 0.7342, Accuracy: 0.6985, Referable Sensitivity: 0.7886, Specificity: 0.9264).
  3. `GET /api/demo-cases` — Calls `screening_service.get_demo_cases()`.
  4. `POST /api/analyze` — Accepts multipart form data (`file` or `demo_id`, `patient_id`, `exam_id`, `eye`). Invokes `screening_service.run_screening(...)`.
  5. `GET /api/download-report/{filename}` — Serves generated HTML clinical report from `webapp/outputs/`.

### 3.2 Frontend Architecture
- **Structure:** Single Page Application (SPA) using vanilla HTML5 (`webapp/templates/index.html`), CSS3 (`webapp/static/css/style.css`), and JavaScript (`webapp/static/js/app.js`). No external frontend frameworks or CDN requirements; fully offline-functional.
- **Components:**
  - Header with system status indicators (Model, Mode, About).
  - Control bar with patient metadata inputs, curated demo dropdown, file upload, and "Analyze Image" action.
  - Image viewport with 5 overlay tabs (`Original`, `Enhanced`, `Vessels`, `Lesions`, `Heatmap`).
  - Right sidebar with 4 collapsible clinical findings panels:
    1. Image Quality Assessment (gauge meter, sub-metrics, status).
    2. DR Severity Classification (headline, referable badge, probability distribution, reference vs AI comparison).
    3. Retinal Landmarks & Lesions (table with OD, fovea, vessel density, MA, EX, HE, SE, NV risk).
    4. Screening Recommendation & Report download actions.

### 3.3 Pipeline Orchestration
- Master coordinator: `webapp/services/screening_service.py` (`ScreeningService.run_screening`).
- **Runtime Flow:**
  1. Image ingestion: OpenCV `cv2.imread` (BGR) and PIL `Image.open` (RGB).
  2. **Stage 1 — Quality Gate:** `QualityService.assess(img_bgr)`.
  3. **Stage 1b — Safety Gate:** If `not quality["gradable"]`, the pipeline stops downstream execution immediately. It does not run enhancement, landmarks, lesion detection, or DL grading. Returns `screening_status = "UNGRADABLE"` with recapture recommendations, generates an ungradable report, and returns early.
  4. **Stage 2 — Enhancement:** `EnhancementService.enhance(img_bgr)` (border crop, 512x512 resize, green CLAHE, slight red/blue CLAHE, Gaussian blur).
  5. **Stage 3 — Anatomical Landmarks:** `StructureService.detect(enhanced_bgr)` (optic disc localization, fovea temporal geometric estimation, vessel morphology, vessel density %).
  6. **Stage 4 — Lesion Analysis:** `LesionService.analyze(enhanced_bgr, od_mask, vessel_density)` (attempts AI Dual-Head U-Net; falls back to classical CV candidate detection).
  7. **Stage 5 — DR Severity Grading & Grad-CAM:** `GradingService.grade(pil_img)` (forward pass via ONNX `best_model.onnx` or PyTorch fallback `best_model.pt`; generates Grad-CAM via `GradCAM`).
  8. **Stage 6 — Result Assembly:** Compiles dictionary with URLs and metrics.
  9. **Stage 7 — Report Generation:** `ReportService.generate(result)` creates standalone HTML report.

### 3.4 Image Quality Gate
- Located in `webapp/services/quality_service.py` (Python) and `core/quality/assessImageQuality.m` (MATLAB).
- Metrics evaluated:
  - **Sharpness:** Laplacian variance of green channel $\mathrm{Var}(\nabla^2 I_G)$. Cutoff: $< 35.0$ indicates severe blur.
  - **Illumination:** Mean intensity of green channel $\mu_G$. Cutoffs: $< 30.0$ severe underexposure, $> 225.0$ severe overexposure/glare.
  - **Contrast:** Standard deviation of active retinal pixels $\sigma_G$.
  - **Field of View (FOV):** Percentage of non-black sensor area ($I_G > 15$). Cutoff: $< 25\%$ fails.
  - **SNR proxy:** Signal-to-noise estimation based on median absolute deviation.
- Composite score: $0.45 \cdot \text{Sharpness} + 0.25 \cdot \text{Illumination} + 0.20 \cdot \text{Contrast} + 0.10 \cdot \text{FOV}$.
- Configured Prototype Thresholds:
  - Score $< 40.0$ or any critical failure $\rightarrow$ `UNGRADABLE` (`gradable = False`).
  - Score $40.0 \le s < 60.0 \rightarrow$ `BORDERLINE` (`gradable = True`).
  - Score $\ge 60.0 \rightarrow$ `GOOD` (`gradable = True`).

### 3.5 Anatomical Landmark Detection
- Located in `webapp/services/structure_service.py`.
- **Optic Disc (OD):** Gaussian blur ($\sigma=31$), peak luminance search (`cv2.minMaxLoc`), radius set to $0.08 \cdot \min(H, W)$. Generates circular binary mask `od_mask` used by lesion detectors to prevent false-positive exudate calls on the physiological disc.
- **Fovea Centralis:** Anatomical temporal projection at $2.5 \times \text{OD radius}$ along horizontal axis, refined by local luminance minimum in ROI.
- **Retinal Vasculature:** Multi-scale black top-hat morphological filtering (11x11 and 5x5 elliptical structuring elements) on green channel, adaptive thresholding, masked to active retina. Calculates vascular density percentage.

### 3.6 Lesion Candidate Detection
- Located in `webapp/services/lesion_service.py` and `python/inference/segment_lesions.py`.
- **Primary AI Engine:** `models/experiments/lesion_segmentation/lesion_unet.onnx` (Dual-Head U-Net trained on IDRiD):
  - Head 1 (Lesions): 4 channels — Microaneurysms (MA), Haemorrhages (HE), Hard Exudates (EX), Soft Exudates (SE).
  - Head 2 (Anatomy): 1 channel — Optic Disc (OD).
- **Heuristic Fallback Engine:**
  - Hard exudates: Bright regions ($I > 175$) excluding dilated optic disc mask.
  - Hemorrhages: Dark lesions on inverted green channel ($255 - I_G > 195$) excluding vessel mask.
  - Microaneurysms: Morphological top-hat small peaks on green channel.
  - Neovascularization risk: Vessel density stratification ($>16\%$ High, $>13.5\%$ Moderate, else Low).

### 3.7 Deep Learning DR Model & Inference Paths
- **Backbone:** EfficientNet-B0 (EXP-001).
- **Target classes:** 5 ICDR severity levels:
  - 0: No DR
  - 1: Mild DR
  - 2: Moderate DR
  - 3: Severe DR
  - 4: Proliferative DR
- **Referral threshold:** Grade $\ge 2$ triggers Referable DR recommendation.
- **Runtime Inference:**
  - **Primary Fast Path:** ONNX Runtime (`models/aptos_efficientnet/best_model.onnx`). Execution time ~35–50 ms on CPU.
  - **PyTorch Fallback Path:** PyTorch module (`models/aptos_efficientnet/best_model.pt`) loaded onto CPU/CUDA via `python/models/efficientnet_dr.py`.
  - **Numerical Parity:** Validated $< 10^{-4}$ max absolute probability difference between PyTorch and ONNX.

### 3.8 Grad-CAM & Layer Inspection
- Located in `python/explainability/gradcam.py`.
- **Target Layer Resolution:**
  - Examined `_find_last_conv(model)`:
    ```python
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv
    ```
  - **Audit Finding on Target Layer:** It dynamically identifies the last `nn.Conv2d` layer rather than relying on a hardcoded string. In EfficientNet-B0 (`torchvision`), this resolves to `backbone[0][8][0]` (the 1x1 point-wise expansion convolution in the final MBConv block, outputting 1280 feature channels).
  - **Audit Finding on Fallbacks & Weaknesses:**
    1. Lines 226–251 contain `_placeholder_heatmap(...)`, which synthesizes an artificial 2D Gaussian blob when the model is unavailable or encounters an error. **This violates Phase 5 requirement ("Never draw synthetic/fake heatmaps. If XAI fails, explicitly report XAI unavailable, never substitute a fake image")**.
    2. Line 244 uses `matplotlib.cm.get_cmap("jet")`, which is removed in Matplotlib 3.11+, causing an unhandled `AttributeError`.
    3. Grad-CAM++ is currently not implemented (only standard first-order Grad-CAM is implemented).
    4. There is no on-demand region masking / sensitivity analysis.
    5. There is no dynamic selection of target class (it always targets `argmax`).

---

## 4. Model & Runtime Artifact Inventory

| Artifact Path | Format | Size | Present | Status / Verification |
|---|---|---|---|---|
| `models/aptos_efficientnet/best_model.onnx` | ONNX | 710 KB | YES | Validated (EXP-001 graph) |
| `models/aptos_efficientnet/best_model.onnx.data` | Binary | 16.0 MB | YES | Validated (EXP-001 weights) |
| `models/aptos_efficientnet/best_model.pt` | PyTorch | 16.3 MB | YES | Validated (state dict, 4,013,953 params) |
| `models/aptos_efficientnet/class_mapping.json` | JSON | 105 B | YES | Validated (5 classes 0–4) |
| `models/aptos_efficientnet/metadata.json` | JSON | 644 B | YES | Validated (QWK=0.7342, Acc=0.6985) |
| `models/aptos_efficientnet/metrics.json` | JSON | 1,018 B | YES | Validated |
| `models/experiments/lesion_segmentation/lesion_unet.onnx` | ONNX | 13.4 MB | YES | Validated (Dual-Head U-Net ONNX) |
| `models/experiments/lesion_segmentation/best_model.pt` | PyTorch | N/A | **NO** | Missing (Causes test failure in `test_lesion_segmentation.py`) |
| `demo/sample_images/demo_grade0.png` | PNG | 79.4 KB | YES | Validated (Grade 0 normal scan) |
| `demo/sample_images/demo_grade1.png` | PNG | 64.6 KB | YES | Validated (Grade 1 mild NPDR) |
| `demo/sample_images/demo_grade2.png` | PNG | 71.0 KB | YES | Validated (Grade 2 moderate NPDR) |
| `demo/sample_images/demo_grade3.png` | PNG | 86.6 KB | YES | Validated (Grade 3 severe NPDR) |
| `demo/sample_images/demo_grade4.png` | PNG | 91.0 KB | YES | Validated (Grade 4 proliferative DR) |
| `demo/sample_images/demo_case6_ungradable.png` | PNG | N/A | **NO** | Missing from disk; `getDemoCases.m` & `test_live_server.py` reference Case 6 |

---

## 5. Baseline Test Execution Results (Pre-Modification)

The baseline test suites were executed using the pinned runtime Python 3.11 (`py -3.11`) without modifying any test or source files.

### Test Suite 1: `webapp/test_webapp.py`
- **Command:** `py -3.11 -m unittest webapp/test_webapp.py`
- **Result:** **PASSED** (16/16 tests passed in 6.539s)
  - `test_01_health_endpoint`: PASS
  - `test_02_homepage_loads`: PASS
  - `test_03_demo_cases_list`: PASS
  - `test_04_invalid_file_extension`: PASS
  - `test_05_empty_file_upload`: PASS
  - `test_06_unsupported_image_content`: PASS
  - `test_07_quality_assessment_good`: PASS
  - `test_08_quality_assessment_ungradable`: PASS
  - `test_09_safety_gate_halts_downstream`: PASS
  - `test_10_enhancement_service`: PASS
  - `test_11_structure_service`: PASS
  - `test_12_lesion_service`: PASS
  - `test_13_grading_service_exp001`: PASS
  - `test_14_gradcam_generation`: PASS
  - `test_15_report_generation`: PASS
  - `test_16_full_end_to_end_screening`: PASS

### Test Suite 2: `python/test_end_to_end_sih.py`
- **Command:** `py -3.11 -m unittest python/test_end_to_end_sih.py`
- **Result:** **PASSED** (18/18 tests passed in 4.329s)
  - All 18 functional requirements (quality metrics, ungradable halt, CLAHE, OD/fovea/vessels, lesion candidates, EXP-001 PyTorch & ONNX parity $< 10^{-4}$, Grad-CAM, HTML report, error handling, CPU latency $< 2.5\text{s}$) passed.

### Test Suite 3: `tests/test_lesion_segmentation.py`
- **Command:** `py -3.11 -m unittest tests/test_lesion_segmentation.py`
- **Result:** **FAILED** (8 passed/skipped, 1 failure)
  - **Failure:** `test_04_pytorch_checkpoint_presence`
  - **Traceback:** `AssertionError: False is not true : PyTorch checkpoint missing at D:\RetinaGuard-main\RetinaGuard-main\models\experiments\lesion_segmentation\best_model.pt`
  - **Root Cause:** The lesion segmentation directory contains the exported ONNX model (`lesion_unet.onnx`), which passes all inference and benchmark tests, but the author did not commit the redundant PyTorch checkpoint `.pt` file to Git.

### Test Suite 4: `python/verify_pipeline.py`
- **Command:** `py -3.11 python/verify_pipeline.py`
- **Result:** **FAILED** (7/9 passed, 2 failures)
  - **Failure 1:** `Grad-CAM: module 'matplotlib.cm' has no attribute 'get_cmap'`
    - **Root Cause:** In `python/explainability/gradcam.py` line 244 (`_placeholder_heatmap`), `matplotlib.cm.get_cmap("jet")` was used. In `matplotlib >= 3.9/3.11`, `cm.get_cmap` is removed in favor of `matplotlib.colormaps["jet"]`.
  - **Failure 2:** `Class weights: No module named 'sklearn'`
    - **Root Cause:** Script attempts `from sklearn.utils.class_weight import compute_class_weight`, but `scikit-learn` is not installed in the Python 3.11 environment.

### Test Suite 5: `python/verify_walkthrough_cases.py`
- **Command:** `py -3.11 python/verify_walkthrough_cases.py`
- **Result:** **PASSED**
  - Evaluated all 5 demo scans (Grades 0–4) through the complete pipeline.
  - Evaluated 7 error modes (corrupt files, 8x8 px, all-zero dark, all-250 glare, severe Gaussian blur).
  - All ungradable tests correctly halted downstream execution.

---

## 6. Actual Working Features vs. Merely Documented

| Feature Area | Documented Claim | Actual Implementation Status | Technical Reality |
|---|---|---|---|
| **DR Backbone Model** | EfficientNet-B0 (EXP-001) trained with Focal Loss on APTOS | **FULLY WORKING** | `best_model.pt` and `best_model.onnx` exist and yield identical predictions (QWK 0.7342). |
| **ONNX Inference** | Fast CPU inference $< 100\text{ms}$ | **FULLY WORKING** | ONNX Runtime session initialized on CPU, runs in ~35 ms. |
| **Quality Assessment** | Laplacian, illumination, contrast, FOV | **FULLY WORKING** | Fully implemented in `QualityService.assess`, accurately separates gradable vs ungradable scans. |
| **Safety Gate** | Halts downstream grading on ungradable images | **FULLY WORKING** | Verified by unit tests; grading and lesion detection return `None` when image is ungradable. |
| **Landmark Detection** | Optic Disc, Fovea, Vessels | **FULLY WORKING** | Classical morphology and luminance extrema accurately identify landmarks and calculate vessel density. |
| **Lesion Segmentation** | Dual-Head U-Net ONNX | **WORKING VIA ONNX** | `lesion_unet.onnx` runs and segments MA, HE, EX, SE, and OD. PyTorch `.pt` is absent. |
| **Lesion Terminology** | "Confirmed lesions" in some docs / UI | **DISCREPANCY** | Currently labeled "AI-detected lesion region" in AI mode, but without independent multi-center validation must be reframed as "Candidate Lesion Evidence". |
| **Grad-CAM** | True Grad-CAM on last conv layer | **WORKING WITH DEFICIENCIES** | Hooks work on last Conv2d layer; however, fallback produces synthetic Gaussian blobs, and Grad-CAM++ is absent. |
| **Integrated Gradients** | Advanced XAI | **MERELY DOCUMENTED** | Not implemented in Python codebase. |
| **Region Sensitivity** | Perturbation / masking analysis | **NOT IMPLEMENTED** | Documented in concept, but no code exists. |
| **ExplainAI** | Grounded explanation engine | **PARTIALLY DOCUMENTED / AD-HOC** | UI displays individual notes, but no unified, structured offline reasoning module exists. |
| **Demo Cases** | 6 Curated Cases | **5 ON DISK, 1 MISSING** | Grades 0–4 present; Case 6 (Ungradable) defined in `getDemoCases.m` but image file is missing from `demo/sample_images/`. |
| **Shared Result Contract** | Unified result object across stack | **PARTIAL** | Screening service returns a dict, but fields vary between gradable and ungradable, and lack strict schema validation. |
| **MATLAB Stack** | Dual-stack execution | **PRESERVED** | All `.m` scripts and App Designer code intact. |

---

## 7. Recommended Modification Points for RetinaGuard 2.0

To fulfill the requirements of **Phases 1 through 20**, the following minimal, surgical modification points are identified:

1. **Central Analysis Contract (`core/contracts/analysis_result.py` or `python/contracts/analysis_result.py`):**
   - Define a single immutable/typed data structure containing: `metadata`, `input`, `quality`, `preprocessing`, `anatomy`, `lesion_candidates`, `model`, `xai`, `region_sensitivity`, `explanation`, `screening`, `timings`, `warnings`.
   - Used identically across backend, UI API, report generation, and tests.

2. **Quality Gate Hardening (`webapp/services/quality_service.py`):**
   - Explicitly document prototype thresholds.
   - Ensure clean failure modes (`UNGRADABLE`, `INVALID_IMAGE`) with structured recapture instructions and zero classifier execution.

3. **Processing Transparency Registry (`python/transparency/registry.py`):**
   - Register all 9 pipeline stages:
     1. Image Quality Gate
     2. Green-channel processing
     3. CLAHE enhancement
     4. Anatomical analysis
     5. Lesion candidate analysis
     6. DR grading
     7. Explainability
     8. Screening interpretation
     9. Report generation
   - Provide for each stage: `name`, `what_it_does`, `why_it_is_used`, `parameters`, `output`, `limitation`.

4. **XAI Engine Upgrade (`python/explainability/gradcam.py` & `gradcam_plus_plus.py`):**
   - Implement **Grad-CAM++** using second- and third-order gradients for superior multi-focal lesion localization.
   - Retain standard **Grad-CAM** as fallback.
   - Remove `_placeholder_heatmap` completely — never emit synthetic heatmaps; if attribution fails, explicitly set `status = "XAI_UNAVAILABLE"`.
   - Allow dynamic target class selection (default: predicted class).
   - Validate map: non-negative, non-empty, finite numbers, correctly sized.

5. **Region Sensitivity Masking (`python/explainability/region_sensitivity.py`):**
   - Implement on-demand sensitivity analysis: mask up to 3 candidate lesion regions, re-run EXP-001 model, and report exact probability delta using neutral terminology ("Model sensitivity to region masking").

6. **Lesion Candidate Reframing (`webapp/services/lesion_service.py`):**
   - Explicitly rename outputs to "candidate regions" (Microaneurysm candidates, Hard Exudate candidates, Hemorrhage candidates).
   - Generate bounding boxes and crops without fabricated confidence scores.

7. **ExplainAI Module (`python/explainability/explain_ai.py`):**
   - Grounded, deterministic, offline rule-based explanation engine answering:
     1. Why was this image accepted/rejected?
     2. Why this preprocessing?
     3. Why this grade?
     4. Which regions influenced the model?
     5. What are the candidate lesions?
     6. How certain/uncertain is the model?
     7. What are known limitations?
   - Purely references structured facts from `AnalysisResult`. Zero hallucination, zero cloud dependency.

8. **Demo Mode Enhancement (`demo/sample_images/` & `screening_service.py`):**
   - Add authentic poor-quality sample `demo_ungradable.png` as Case 6 so all 6 demo cases function seamlessly.
   - Map 6 demo cases: Normal (Grade 0), Mild (Grade 1), Moderate (Grade 2), Severe (Grade 3), Proliferative (Grade 4), Poor-Quality (Ungradable).

9. **Frontend Workstation Evolution (`webapp/templates/index.html`, `webapp/static/js/app.js`, `style.css`):**
   - Update UI to clinical workstation aesthetics:
     - Home / Case Selection
     - Analysis Workspace (with "Why?" transparency modals for each stage)
     - Result Summary (clear 5-class progression display, uncalibrated model probabilities, margin)
     - Evidence Workspace (crops, Grad-CAM++, region sensitivity on-demand)
     - ExplainAI Panel (grounded questions & answers)
     - Clinical Report & Verification View

10. **Report Alignment (`webapp/services/report_service.py`):**
    - Populate report strictly from `AnalysisResult` so UI, JSON, and HTML report never diverge.

---

## 8. Files That Must NOT Be Replaced

To guarantee backward compatibility and protect the core research assets, the following files **must NOT be replaced, deleted, or fundamentally restructured**:

1. `app/RetinaGuardApp.m` — MATLAB App Designer clinical application
2. `launch.m` — MATLAB launch entry point
3. `runAllTests.m` — MATLAB test runner
4. `core/*` — All MATLAB pipeline functions (`assessImageQuality.m`, `enhanceRetinalImage.m`, `detectStructures.m`, `detectLesions.m`, `gradeDR.m`, etc.)
5. `models/aptos_efficientnet/best_model.pt` — PyTorch EXP-001 trained weights
6. `models/aptos_efficientnet/best_model.onnx` & `best_model.onnx.data` — Production ONNX model
7. `models/aptos_efficientnet/class_mapping.json` & `metadata.json` — Official model cards and mappings
8. `models/experiments/lesion_segmentation/lesion_unet.onnx` — Trained lesion segmentation network
9. `demo/sample_images/demo_grade0.png` through `demo_grade4.png` — Curated fundus demo cases

---

## 9. Conclusion & Read-out

The existing RetinaGuard repository is solid, with passing core end-to-end tests and working model inference. The identified gaps (Matplotlib cm deprecation, missing lesion segmentation `.pt` checkpoint file, placeholder Gaussian heatmap fallback, lack of Grad-CAM++, lack of region sensitivity, lack of grounded ExplainAI, and missing Case 6 image) will be systematically resolved in the RetinaGuard 2.0 implementation without breaking the existing core or MATLAB stack.

Audit completed. Awaiting user approval of implementation plan before proceeding to Phase 1.
