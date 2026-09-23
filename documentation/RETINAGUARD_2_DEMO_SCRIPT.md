# RetinaGuard 2.0 — Hackathon Demonstration Script & Evaluation Protocol
**SIH 2026 Problem Statement: SIH26038**  
**Team: VyuhaX**  
**Workstation URL: `http://localhost:8000`**  

---

## 1. Demonstration Setup & Prerequisites

Before presenting to the evaluation panel, verify the system is running locally:

```bash
# In Windows PowerShell / Terminal:
cd d:\RetinaGuard-main\RetinaGuard-main
py -3.11 webapp/run.py
```
Open Chrome or Edge at: **`http://localhost:8000`**

### Key Evaluation Themes to Emphasize:
1. **Explainability Over Black-Box**: Not just a classification grade, but an audited 12-stage transparent pipeline.
2. **Clinical Safety First**: Automated quality gate stops ungradable images to prevent medical misdiagnosis.
3. **Evidence-Grounded**: Candidate lesion crops and deterministic ExplainAI reasoning based strictly on real facts.
4. **Honest Attribution**: Real Grad-CAM++ with higher-order gradients; zero synthetic/fake heatmaps.
5. **Completely Offline Edge**: Sub-400ms inference on standard CPU with zero cloud dependencies.

---

## 2. Step-by-Step Jury Walkthrough (7-Minute Presentation)

### Phase 1: Clinical Workstation Overview (1 Minute)
- **Action**: Direct the jury's attention to the top navigation header and the **8-Stage Pipeline Stepper**:
  - Point out the **Backbone tag**: `EXP-001 (EfficientNet-B0)`.
  - Point out the **Mode tag**: `OFFLINE EDGE`.
  - Click the **"Pipeline Transparency"** button in the header. Show how every single filter (Border removal, CLAHE, Top-hat, etc.) exposes what it does, why it is used, actual parameters, and clinical limitations.
- **Narrative**:
  > *"Respected Evaluators, in clinical ophthalmology, a black-box AI that simply outputs 'Grade 2' cannot be trusted. RetinaGuard 2.0 transforms AI from a black box into a transparent, evidence-centered screening workstation where every step is explainable."*

---

### Phase 2: Safety Gate Demonstration — Case 6 (1.5 Minutes)
- **Action**:
  1. In the **Choose Demo Grade** dropdown, select: **`Case 6 — Poor Quality (Ungradable)`**.
  2. Click **`Analyze Image ▶`**.
- **Observations to Highlight**:
  - **Quality Score**: Composite score drops to **14.1 / 100** (`UNGRADABLE`).
  - **Laplacian Variance**: Sharpness is measured at **1.2** (far below the required threshold of 35.0).
  - **Red Alert Banner**: *"IMAGE UNGRADABLE: Downstream grading stopped for patient safety."*
  - **Grading & Lesions Panels**: Immediately halted (`UNGRADABLE`, Grade -1). No false probabilities or simulated lesions are fabricated.
  - **ExplainAI Accordion**: Automatically generates Question 1 explaining *why the image was rejected* and actionable instructions for the camera operator (*"Recapture with improved focus and full pupil dilation"*).
- **Narrative**:
  > *"Here is our first critical innovation: the Patient Safety Gate. An ungradable scan with severe motion blur or pupil vignetting is the number one cause of AI false negatives in the field. Rather than hallucinating a false 'No DR' diagnosis, RetinaGuard 2.0 halts downstream inference, protects the patient, and gives the operator immediate recapture instructions."*

---

### Phase 3: Normal Retina Screening — Case 1 (1 Minute)
- **Action**:
  1. Select **`Case 1 — Normal Retina (No DR)`**.
  2. Click **`Analyze Image ▶`**.
- **Observations to Highlight**:
  - **Quality Gate**: Passes with Score **76.3 / 100** (`GOOD`).
  - **Anatomy**: Optic Disc detected, Fovea projected, Vessel density calculated at **14.56%**.
  - **EXP-001 Classification**: Grade 0 — No DR (**90.0% model probability**).
  - **ICDR Progression Scale**: Marker '0' lights up in green.
  - **Triage**: Non-referable (`NON-REFERABLE (NO)`), recommendation: Routine follow-up in 12 months.
  - **Attribution**: Toggle **`Model Attribution (Grad-CAM++)`** to show smooth, diffuse receptive field attention over healthy background tissue.

---

### Phase 4: Referable Pathology & Candidate Crops — Case 3 (1.5 Minutes)
- **Action**:
  1. Select **`Case 3 — Moderate NPDR`**.
  2. Click **`Analyze Image ▶`**.
- **Observations to Highlight**:
  - **DR Severity**: Grade 2 — Moderate DR (**40.0% model probability**).
  - **ICDR Progression Scale**: Marker '2' lights up in amber.
  - **Decision Margin Card**: Shows **11.9% margin over runner-up** (Grade 3 — Severe DR).
  - **Calibration Notice**: *"Calibration: not applied (raw model probabilities)"*.
  - **Referral Badge**: `REFERABLE DR (YES)` triggered by the configured prototype screening rule (Grade $\ge 2$).
  - **Candidate Lesion Evidence**:
    - Microaneurysm candidates and hemorrhage candidates identified.
    - Scroll down the left column to the **Localized Candidate Evidence Crops Gallery**: show authentic $96 \times 96$ cropped cutouts of candidate lesions with bounding boxes.
- **Narrative**:
  > *"Notice our clinical terminology: we present these findings as candidate evidence for human review, never as autonomous confirmed lesions. The screener can visually inspect the actual localized crops to confirm the AI's evidence before referring the patient."*

---

### Phase 5: On-Demand Region Sensitivity & Dynamic XAI (1.5 Minutes)
- **Action**:
  1. While still on Case 3, locate the **On-Demand Region Masking Analysis** box below the image.
  2. Click **`Test Model Sensitivity`**.
- **Observations to Highlight**:
  - The system temporarily obscures candidate lesion regions and re-runs the **exact same EXP-001 model forward pass** without modifying weights.
  - Displays a clean perturbation table showing: Original Probability, Masked Probability, Probability Delta, and neutral statement: *"Model sensitivity to region masking: Masking this region reduced the selected class probability by 2.4 percentage points."*
  - Emphasize the neutral terminology: this proves model sensitivity, **not** causal biological proof.
  3. Now switch to the **Heatmap layer** and use the **Target Class dropdown**:
     - Switch from *Predicted Class (Moderate DR)* to *Grade 4 (Proliferative DR)*.
     - Observe how the attribution heatmap shifts to investigate counter-evidence.

---

### Phase 6: Grounded ExplainAI & Report Parity (1 Minute)
- **Action**:
  1. Scroll through Panel 4: **`ExplainAI — Grounded Reasoning`**.
  2. Expand Questions 1 through 8 to show deterministic, offline answers referencing actual numbers (quality score, candidate counts, probability margin, rule description).
  3. In Panel 5, click **`View Full HTML Report ↗`**.
  4. Compare the report side-by-side with the UI:
     - Grade 2, 40.0% confidence, candidate counts, decision margin, and disclaimers match with 100% parity.
- **Closing Narrative**:
  > *"Every value in our UI, our ExplainAI reasoning engine, and our printable clinical report comes from one unified contract: AnalysisResult. There is zero hallucination, zero cloud dependency, and total transparency. RetinaGuard 2.0 is ready for deployment in rural community healthcare."*

---

## 3. Quick Reference Matrix for Demo Cases

| Case | Reference Condition | Expected Model Output | Key Demo Objective |
| :---: | :--- | :--- | :--- |
| **Case 1** | Normal Retina | Grade 0 — No DR | Demonstrates healthy baseline, anatomical localization, routine triage. |
| **Case 2** | Mild NPDR | Grade 1 — Mild DR | Demonstrates isolated microaneurysm candidates, non-referable triage. |
| **Case 3** | Moderate NPDR | Grade 2 — Moderate DR | Demonstrates referable triage threshold, candidate crops gallery, decision margin. |
| **Case 4** | Severe NPDR | Grade 3 — Severe DR | Demonstrates extensive candidate hemorrhages, urgent referral timeline. |
| **Case 5** | Proliferative DR | Grade 4 — Proliferative DR | Demonstrates advanced DR, high neovascularization risk indicator. |
| **Case 6** | Poor Optical Quality | **UNGRADABLE (Halted)** | **CRITICAL SAFETY DEMO**: Proves automated quality gate halts grading on blur/underexposure. |
