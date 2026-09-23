# RetinaGuard 2.0 — Clinical, Algorithmic & Operational Limitations
**SIH 2026 Problem Statement: SIH26038**  
**Team: VyuhaX**  
**Scope: Regulatory Transparency & Safety Boundaries**  

---

## 1. Mandatory Screening Language & Prohibited Terminology

To guarantee medical safety and prevent clinical over-reliance on artificial intelligence, RetinaGuard 2.0 establishes strict linguistic guidelines across all user interfaces, API documentation, exported reports, and research papers:

### 1.1 Prohibited Claims:
The following terms and concepts are **strictly prohibited** in RetinaGuard 2.0:
- ❌ *"Clinical-grade diagnosis"* or *"Specialist replacement"*
- ❌ *"Definitive / autonomous diagnosis"*
- ❌ *"Confirmed lesion segmentation"*
- ❌ *"Guaranteed accuracy"* or *"Zero-error AI"*
- ❌ *"Zero-hallucination"* (replaced by *"deterministic, evidence-grounded"*)
- ❌ *"Causal proof"* in attribution or perturbation heatmaps

### 1.2 Approved Objective Terminology:
The system consistently employs standard scientific screening nomenclature:
- ✅ **"AI-assisted screening decision support"**
- ✅ **"Model prediction"**
- ✅ **"Candidate evidence"** (e.g. *Microaneurysm candidates*, *Hard exudate candidates*)
- ✅ **"Screening interpretation & triage"**
- ✅ **"Configured prototype screening rule"**
- ✅ **"Deterministic, evidence-grounded explanation"**
- ✅ **"Professional ophthalmic evaluation recommended"**
- ✅ **"Research prototype"**

---

## 2. Clinical & Algorithmic Limitations

### 2.1 Candidate Lesion Evidence vs Confirmed Clinical Lesions
- **Computer Vision Candidates**: Features labeled as *Microaneurysm candidates* or *Hard exudate candidates* are extracted using deep neural segmentation (IDRiD U-Net) or morphological top-hat filtering. 
- **Confounders**: Normal retinal variants (e.g., crossing vessels, choroidal pigment patches, small drusen, or reflex artifacts) can mimic microvascular lesions.
- **Clinical Rule**: Candidate counts provide visual screening evidence to guide the screener; they **do not** constitute biopsy- or specialist-confirmed clinical pathology.

### 2.2 Neovascularization Detection & Vessel Density Heuristics
- **Algorithmic Nature**: The Neovascularization (NV) Risk Indicator calculates global vascular density and peri-papillary vessel branching using classical multi-scale 2D Gabor and ridge filters.
- **Severe Limitation**: True clinical neovascularization of the disc (NVD) or retina elsewhere (NVE) consists of fragile, irregular, fine microvessels that frequently require **Fluorescein Angiography (FFA)** or **Optical Coherence Tomography Angiography (OCTA)** to definitively confirm active leakage.
- **Clinical Directive**: Elevated vessel density is a **heuristic risk indicator only**. It must never be interpreted as confirmed proliferative neovascularization in the absence of specialist clinical slit-lamp biomicroscopy.

### 2.3 Optical Quality Gate Boundaries
- **Optical Proxies**: The Image Quality Assessment Gate evaluates composite optical metrics (Laplacian sharpness variance, illumination mean, active contrast, and circular field of view).
- **Limitation**: These metrics evaluate global optical clarity across the central 45-degree field. Subtle focal artifacts (e.g. dust on camera objective lens or minor corneal reflections) may produce a "GOOD" score while partially obscuring a peripheral quadrant. Conversely, a patient with high myopia or dark fundus pigmentation may trigger a "BORDERLINE" illumination warning despite the macula being clinically gradable.

---

## 3. Data & Generalization Boundaries

### 3.1 Training Distribution
- **Dataset Origin**: Trained primarily on the APTOS 2019 Blindness Detection dataset (scanned at Aravind Eye Hospital, India) and IDRiD (Indian Diabetic Retinopathy Image Dataset).
- **Demographic Bias**: The majority of training scans originate from South Asian eyes with darker retinal pigmentation (higher melanin content in retinal pigment epithelium). Performance on Caucasian, African, or East Asian fundus variations may exhibit subtle distribution shifts.

### 3.2 Fundus Camera Variance
- **Tested Hardware**: Validated on images captured with standard desktop fundus cameras (e.g., Zeiss, Topcon, Canon non-mydriatic cameras).
- **Smartphone & Handheld Cameras**: Handheld smartphone fundus adapters (e.g., Remidio, Volk iNview) introduce non-uniform radial distortion, chromatic aberration, and illumination falloff that may degrade optical quality metrics and lesion candidate detection unless specifically re-calibrated.

---

## 4. Operational & Deployment Considerations

### 4.1 Calibration Status
- **Raw Probabilities**: Softmax probabilities reflect raw model outputs ($z_i = \frac{e^{y_i}}{\sum e^{y_j}}$).
- **Calibration Notice**: Stated explicitly as `"Calibration: not applied (raw model probabilities)"`. The system does not claim calibrated Bayesian posterior probabilities because post-hoc calibration (e.g. temperature scaling or isotonic regression) has not been validated across diverse real-world clinic cameras.

### 4.2 Screener Qualification
- **User Profile**: RetinaGuard 2.0 is designed for trained community health workers, ophthalmic nurses, and primary care optometrists.
- **Operator Requirement**: Operators must be trained to recognize ungradable scans, verify pupil dilation, and inspect candidate crops before transmitting cases to referral centers.

---

## 5. Regulatory & Ethical Disclaimer

```
IMPORTANT CLINICAL DISCLAIMER:
RetinaGuard 2.0 is an AI-assisted diabetic retinopathy screening decision-support 
research prototype. It does NOT provide an autonomous medical diagnosis. All findings, 
candidate lesion counts, and recommendations must be confirmed by a licensed ophthalmologist 
or retina specialist before clinical management or therapeutic interventions are undertaken.
```
