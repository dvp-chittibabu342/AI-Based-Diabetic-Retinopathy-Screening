# RetinaGuard 2.0 — Transparent, Evidence-Centered Explainable AI-Assisted DR Screening Workstation

> **SIH 2026 Problem Statement: SIH26038** | AI-Assisted Retinal Image Analysis Workstation  
> **Team: VyuhaX**  
> *Edge-deployable screening workstation integrating deep learning classification, computer vision candidate evidence, real Grad-CAM++ attribution, on-demand region sensitivity, and deterministic ExplainAI reasoning.*

---

## 1. Project Overview

**RetinaGuard 2.0** is an offline-capable, evidence-grounded AI screening decision-support research prototype engineered for early detection, anatomical assessment, and severity grading of Diabetic Retinopathy (DR) from digital color fundus photographs.

Designed for deployment in primary healthcare centers, rural community clinics, and mobile tele-ophthalmology screening vans, RetinaGuard 2.0 transforms AI from a black-box classifier into a transparent, multi-stage clinical screening workstation where:
- Every image filter and preprocessing stage is documented and explainable via interactive "Why?" transparency modals.
- Optical quality failures immediately halt downstream grading to prevent medical misdiagnosis.
- Microvascular abnormalities are presented as **candidate evidence** with inspectable visual crops, never as autonomous confirmed lesions.
- Neural decisions are localized using **real Grad-CAM++** attribution maps with zero synthetic/fake heatmaps.
- Model sensitivity is verifiable on-demand through candidate region masking perturbations.
- Healthcare screeners receive deterministic, evidence-grounded answers to 8 core clinical questions via an offline **ExplainAI** engine.
- All UI elements, REST API responses, and printable clinical reports derive from a single unified contract: `AnalysisResult`.

The system features two operational interfaces:
1. **Clinical Web Workstation**: An interactive, responsive browser interface built on FastAPI, vanilla CSS, and modern web standards.
2. **MATLAB Desktop Workstation**: A standalone clinical App Designer interface supporting native MATLAB workflows.

---

## 2. SIH Problem Statement Mapping (SIH26038)

| SIH Requirement | RetinaGuard 2.0 Implementation | Verification & Evidence |
| :--- | :--- | :--- |
| **Automated Image Quality Assessment** | Multi-metric optical gate analyzing Laplacian sharpness, green-channel illumination, contrast, and circular Field-of-View (FOV). | Automatically halts downstream inference on ungradable scans (< 40.0 score) with actionable recapture guidance. |
| **Image Preprocessing & Enhancement** | Circular aperture border extraction, canonical resolution standardization, and green-channel CLAHE (`clipLimit=2.0`, `tileGrid=8x8`). | Documented in 12-stage pipeline transparency registry with "Why?" UI inspector. |
| **Anatomical Landmark Detection** | Morphological peak localization for Optic Disc (OD), temporal geometric projection for Fovea Centralis, and vascular tree density %. | Fully automated landmark localization and vascular area density quantification. |
| **Candidate Lesion Evidence Detection** | IDRiD-trained dual-head U-Net ONNX model (`lesion_unet.onnx`) with morphological top-hat CV fallback. Extracts $96 \times 96$ px visual crops. | Microaneurysm candidates, Hard exudate candidates, Soft exudate candidates, Hemorrhage candidates, and Neovascularization Risk Indicator. |
| **Multi-Class DR Severity Grading** | 5-class International Clinical Diabetic Retinopathy (ICDR) grading (Grades 0 to 4) using locked production model **EXP-001** (`EfficientNet-B0`). | $\text{QWK} = 0.7342$, $\text{Accuracy} = 69.85\%$, $\text{Macro F1} = 0.4981$ across 733 held-out test scans. |
| **Probability & Margin Transparency** | Full 5-class probability vector, top model probability, runner-up class, decision margin %, and uncalibrated label. | Labeled *"Calibration: not applied (raw model probabilities)"* to prevent over-confidence. |
| **Real Explainable AI (XAI)** | Real Grad-CAM++ using second- and third-order analytical gradients w.r.t. `features.8.0` (1280 channels) with real Grad-CAM fallback. | **Zero synthetic heatmaps**: Returns explicit `XAI_UNAVAILABLE` on model failure. Dynamic target class selection (Grades 0–4). |
| **On-Demand Region Sensitivity** | Evaluates model probability response when masking candidate evidence regions without altering base model weights. | Neutral reporting: *"Model sensitivity to region masking"* with probability delta. |
| **Grounded ExplainAI Engine** | Fully offline, deterministic reasoning engine answering 8 mandatory screening questions strictly using `AnalysisResult` facts. | Zero external LLMs, zero cloud dependencies, mathematically grounded. |
| **Clinical Decision Support & Triage** | Automated screening triage based on configured prototype rule (Grade $\ge 2$ = Referable DR) with urgency timelines. | $\text{Sensitivity} = 78.86\%$, $\text{Specificity} = 92.64\%$. |
| **Auditable Clinical Reporting** | Automated, print-ready, self-contained HTML medical screening report generated exclusively from `AnalysisResult`. | 100% parity between UI display, ExplainAI answers, and exported report. |

---

## 3. 12-Stage Pipeline Architecture

```
                  Raw Color Fundus Photograph
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: Image Quality Assessment Gate                      │
│ (Laplacian Sharpness, Illumination, Contrast, FOV)          │
└──────────────┬───────────────────────────────┬──────────────┘
               │ [Score < 40.0]                │ [Score >= 40.0]
               ▼                               ▼
┌───────────────────────────┐    ┌────────────────────────────┐
│   SAFETY GATE HALT        │    │ STAGE 2: Border Removal    │
│ Downstream grading halted;│    └─────────────┬──────────────┘
│ clinical recapture advice │                  ▼
└──────────────┬────────────┘    ┌────────────────────────────┐
               │                 │ STAGE 3: Scale Standardize │
               │                 └─────────────┬──────────────┘
               │                               ▼
               │                 ┌────────────────────────────┐
               │                 │ STAGE 4: Green-Ch. CLAHE   │
               │                 └─────────────┬──────────────┘
               │                               ▼
               │                 ┌────────────────────────────┐
               │                 │ STAGE 5: Anatomy & Vessels │
               │                 │ Optic Disc, Fovea, Vessels │
               │                 └─────────────┬──────────────┘
               │                               ▼
               │                 ┌────────────────────────────┐
               │                 │ STAGE 6: Candidate Lesions │
               │                 │ U-Net Candidates & Crops   │
               │                 └─────────────┬──────────────┘
               │                               ▼
               │                 ┌────────────────────────────┐
               │                 │ STAGE 7: EXP-001 ONNX      │
               │                 │ 5-Class ICDR Softmax       │
               │                 └─────────────┬──────────────┘
               │                               ▼
               │                 ┌────────────────────────────┐
               │                 │ STAGE 8: Real Grad-CAM++   │
               │                 │ 2nd/3rd-Order Attribution  │
               │                 └─────────────┬──────────────┘
               │                               ▼
               │                 ┌────────────────────────────┐
               │                 │ STAGE 9: Region Masking    │
               │                 │ On-Demand Sensitivity Test │
               │                 └─────────────┬──────────────┘
               │                               ▼
               │                 ┌────────────────────────────┐
               │                 │ STAGE 10: ExplainAI Engine │
               │                 │ 8 Grounded Clinical Answers│
               │                 └─────────────┬──────────────┘
               │                               ▼
               │                 ┌────────────────────────────┐
               │                 │ STAGE 11: Screening Triage │
               │                 │ Prototype Rule: Grade >= 2 │
               │                 └─────────────┬──────────────┘
               │                               ▼
               │                 ┌────────────────────────────┐
               │                 │ STAGE 12: HTML Report Gen  │
               │                 │ Sourced from Single Truth  │
               │                 └─────────────┬──────────────┘
               │                               │
               └───────────────┬───────────────┘
                               │
                               ▼
        ┌─────────────────────────────────────────────┐
        │ UNIFIED CONTRACT: AnalysisResult JSON Schema│
        └─────────────────────────────────────────────┘
```

---

## 4. Key Technical Capabilities

- **Unified Single Source of Truth**: Sourced exclusively from `python/contracts/analysis_contract.py` guaranteeing zero divergence between UI, ExplainAI, and printed reports.
- **Patient Safety Gate**: Automatically stops downstream processing on ungradable images (blur, pupil vignetting, extreme exposure) to eliminate false reassurances.
- **Candidate Evidence Framing**: Presents findings as candidate evidence with bounding boxes and localized $96 \times 96$ crops for screener inspection.
- **Real Grad-CAM++ Attribution**: Uses analytical higher-order gradients to localize multiple scattered microvascular abnormalities without synthetic artifacts.
- **Dynamic Target Class Selection**: Screeners can evaluate model attribution for any class (Grades 0 to 4) to investigate differential evidence.
- **On-Demand Region Sensitivity**: Measures class probability deltas when obscuring candidate lesion areas using exact neutral phrasing (*"Model sensitivity to region masking"*).
- **Offline Grounded ExplainAI**: Answers 8 mandatory screening questions deterministically without cloud dependencies or hallucination risks.
- **100% Offline Edge Runtime**: Sub-400ms end-to-end latency on quad-core CPU hardware; runs anywhere without internet.

---

## 5. System Requirements

### Hardware Requirements
- **Processor**: Intel Core i3 / AMD Ryzen 3 (Quad-Core, 2.0 GHz+) or higher
- **RAM**: Minimum 4 GB RAM (8 GB recommended)
- **Disk Space**: ~2 GB free disk space for model weights, dependencies, and reference cases
- **GPU**: Optional (CPU execution is fully optimized via ONNX Runtime)

### Software Requirements
- **Operating System**: Windows 10/11, macOS, or Linux (Ubuntu 20.04+)
- **Python**: Python 3.10 or 3.11 (Python 3.11 recommended)
- **MATLAB (Optional)**: MATLAB R2021b or newer (only required for running the MATLAB app)

---

## 6. Installation & Quick Start

### 6.1 Clone the Repository
```bash
git clone https://github.com/sailohith95/RetinaGuard.git
cd RetinaGuard
```

### 6.2 Python Virtual Environment Setup
```bash
# Windows
py -3.11 -m venv venv
venv\Scripts\activate

# macOS / Linux
python3.11 -m venv venv
source venv/bin/activate
```

### 6.3 Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 7. Launching RetinaGuard 2.0

### Web Workstation (Recommended)
From the repository root:
```bash
python webapp/run.py
```
- Automatically starts FastAPI server on `http://127.0.0.1:8000`.
- Opens your default web browser to the clinical screening workstation.
- Accessible across local network via `http://<your-ip>:8000`.

### MATLAB Desktop Application
Open MATLAB, navigate to `RetinaGuard`, and execute:
```matlab
launch
```
This initializes paths, checks model weights, and opens the standalone App Designer interface.

---

## 8. Curated Authentic Demo Cases

The system bundles **6 authentic clinical reference cases** for immediate demonstration:

| Case ID | Reference Condition | Reference Grade | Expected Workstation Behavior |
| :---: | :--- | :---: | :--- |
| **Case 1** | Normal Healthy Retina | Grade 0 | Score 76.3 (GOOD), Grade 0 (90.0% prob), Non-referable, 12-month follow-up. |
| **Case 2** | Mild NPDR | Grade 1 | Isolated microaneurysm candidates, non-referable triage (6–12 month follow-up). |
| **Case 3** | Moderate NPDR | Grade 2 | Hard exudate & hemorrhage candidates, triggers Referable DR badge (3–6 month referral). |
| **Case 4** | Severe NPDR | Grade 3 | Multiple candidate hemorrhages, urgent specialist referral recommended (1 month). |
| **Case 5** | Proliferative DR | Grade 4 | Extensive pathology, high neovascularization risk indicator, urgent referral required. |
| **Case 6** | Poor Optical Quality | **Ungradable (-1)** | **Safety Gate Halts Grading**: Score 14.1, Laplacian var 1.2, actionable recapture guidance. |

---

## 9. Verification & Automated Test Suites

RetinaGuard 2.0 includes comprehensive automated test coverage:

```bash
# 1. Run RetinaGuard 2.0 Core Verification Suite (11 tests)
py -3.11 -m unittest webapp/test_retinaguard_2.py

# 2. Run Webapp Integration Suite (16 tests)
py -3.11 -m unittest webapp/test_webapp.py

# 3. Run Lesion U-Net Segmentation Suite (9 tests)
py -3.11 -m unittest tests/test_lesion_segmentation.py

# 4. Run SIH End-to-End Test Suite (18 tests)
py -3.11 -m unittest python/test_end_to_end_sih.py

# 5. Run Python Pipeline Verification
py -3.11 python/verify_pipeline.py

# 6. Run Empirical Walkthrough Evaluation across all 6 cases
py -3.11 python/verify_walkthrough_cases.py
```

---

## 10. Documentation Index

Detailed engineering, clinical, and regulatory specifications are available in `documentation/`:
- [`RETINAGUARD_2_ARCHITECTURE.md`](file:///d:/RetinaGuard-main/RetinaGuard-main/documentation/RETINAGUARD_2_ARCHITECTURE.md): Multi-stage pipeline architecture & `AnalysisResult` contract schema.
- [`RETINAGUARD_2_XAI.md`](file:///d:/RetinaGuard-main/RetinaGuard-main/documentation/RETINAGUARD_2_XAI.md): Grad-CAM++ formulation, higher-order gradients, and region sensitivity.
- [`RETINAGUARD_2_MODEL_CARD.md`](file:///d:/RetinaGuard-main/RetinaGuard-main/documentation/RETINAGUARD_2_MODEL_CARD.md): EXP-001 empirical validation (QWK 0.7342, accuracy 69.85%), training loss, and calibration.
- [`RETINAGUARD_2_LIMITATIONS.md`](file:///d:/RetinaGuard-main/RetinaGuard-main/documentation/RETINAGUARD_2_LIMITATIONS.md): Clinical boundaries, prohibited terms, and regulatory disclaimers.
- [`RETINAGUARD_2_DEMO_SCRIPT.md`](file:///d:/RetinaGuard-main/RetinaGuard-main/documentation/RETINAGUARD_2_DEMO_SCRIPT.md): Step-by-step hackathon jury presentation script.

---

## 11. Clinical Disclaimer

```
IMPORTANT CLINICAL DISCLAIMER:
RetinaGuard 2.0 is an AI-assisted diabetic retinopathy screening decision-support 
research prototype developed for SIH 2026 (PS: SIH26038, Team VyuhaX). 
It does NOT provide an autonomous medical diagnosis. All findings, candidate lesion counts, 
and recommendations must be confirmed by a licensed ophthalmologist or retina specialist 
before clinical management or treatment decisions are undertaken.
```
