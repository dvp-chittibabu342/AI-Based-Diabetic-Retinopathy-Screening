# RetinaGuard 2.0 — Model Card: EXP-001 (EfficientNet-B0)
**SIH 2026 Problem Statement: SIH26038**  
**Team: VyuhaX**  
**Model Identifier: `EXP-001`**  
**Standard: IEEE / ACM Model Card Framework**  

---

## 1. Model Details

### 1.1 Model Overview
- **Model Name**: RetinaGuard EXP-001
- **Architecture**: EfficientNet-B0 (Tan & Le, 2019)
- **Pretrained Weights**: ImageNet-1k transfer learning initialization
- **Input Dimensions**: $3 \times 224 \times 224$ normalized RGB tensor
- **Output Layer**: 5-unit linear classification head with Softmax activation
- **Total Parameters**: 4,013,953 (100% trainable in fine-tuning stage)
- **Deployment Format**: ONNX Runtime (Graph v1.14+, Opset 17) & PyTorch (.pt)
- **License**: Research Prototype for SIH 2026 (Educational & Scientific Use)

### 1.2 Verified Class Mapping (Source of Truth)
Extracted directly from `models/aptos_efficientnet/class_mapping.json`:

| Class Index | Clinical Severity Label (ICDR) | Clinical Definition |
| :---: | :--- | :--- |
| **0** | **No DR** | No microaneurysms or retinal abnormalities |
| **1** | **Mild DR** | Microaneurysms only |
| **2** | **Moderate DR** | More than microaneurysms, but less than severe NPDR (exudates, hemorrhages) |
| **3** | **Severe DR** | >20 intraretinal hemorrhages in 4 quadrants, venous beading in 2+, prominent IRMA in 1+ |
| **4** | **Proliferative DR** | Neovascularization and/or vitreous/preretinal hemorrhage |

---

## 2. Training Data & Methodology

### 2.1 Dataset Composition
- **Primary Training Cohort**: APTOS 2019 Blindness Detection Dataset (Aravind Eye Hospital, India)
- **Lesion Segmentation Cohort**: IDRiD (Indian Diabetic Retinopathy Image Dataset)
- **Cohort Size**: 3,662 diverse fundus camera photographs
- **Train / Validation / Test Split**: 80% Train (2,929 scans) / 20% Held-Out Test (733 scans)

### 2.2 Class Distribution & Class Weighting
Retinal datasets exhibit severe class imbalance, with Grade 0 significantly outnumbering Grade 3 and Grade 4. EXP-001 applied balanced sample weighting during optimization:

| Grade | Severity | Sample Count | Inverse Frequency Weight |
| :---: | :--- | :---: | :---: |
| **0** | No DR | 1,805 | 0.390 |
| **1** | Mild DR | 370 | 1.950 |
| **2** | Moderate DR | 999 | 0.780 |
| **3** | Severe DR | 193 | 3.900 |
| **4** | Proliferative DR | 295 | 2.600 |

### 2.3 Training Hyperparameters & Loss
- **Loss Function**: Multi-Class Focal Loss ($\gamma = 2.0$, $\alpha = \text{class\_weights}$) to suppress gradient contribution from easily classified Grade 0 samples and force convergence on rare severe grades.
- **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $= 10^{-4}$)
- **Learning Rate Schedule**: Cosine Annealing with Warm Restarts ($\text{lr}_{\text{max}} = 3 \times 10^{-4}$, $\text{lr}_{\text{min}} = 10^{-6}$)
- **Data Augmentation**: Random affine rotations ($\pm 180^\circ$), horizontal/vertical flips, subtle color jitter (brightness 0.1, contrast 0.1), and CLAHE green-channel extraction.
- **Hardware**: Single NVIDIA RTX GPU; fine-tuning completed across 25 epochs.

---

## 3. Empirical Evaluation on Held-Out Test Cohort

Tested on **733 independent, held-out fundus photographs** from the APTOS 2019 test split (unseen during training):

### 3.1 Primary Clinical Screening Metrics

| Metric | Empirical Value | Clinical Significance |
| :--- | :---: | :--- |
| **Quadratic Weighted Kappa (QWK)** | **0.7342** | High inter-rater agreement with ophthalmologist consensus (benchmark standard for DR competitions). |
| **Exact 5-Class Accuracy** | **69.85%** | Multi-class exact concordance across all 5 discrete ICDR stages. |
| **Referable DR Sensitivity (Grade $\ge 2$)** | **78.86%** | Ability to identify patients who genuinely require specialist clinical evaluation. |
| **Referable DR Specificity** | **92.64%** | Minimizes unnecessary clinic referrals and false alarms in primary healthcare centers. |
| **Referable DR Negative Predictive Value (NPV)** | **94.81%** | Very high assurance that patients triaged as non-referable do not harbor vision-threatening pathology. |
| **Macro F1-Score** | **0.4981** | Unweighted average F1-score across severely imbalanced minority classes (e.g. Grade 3). |

---

## 4. Probability Output, Decision Margins & Calibration Status

### 4.1 Calibration Transparency: "Calibration: not applied"
Many commercial AI products present raw softmax outputs as calibrated Bayesian probabilities (e.g. claiming "99.2% probability of disease"). Deep neural networks trained with cross-entropy or focal loss are frequently over-confident.

**RetinaGuard 2.0 explicitly labels all raw softmax outputs**:
```
Model probability: 40.0% (Moderate Confidence)
Calibration: not applied (raw model probabilities)
```
Unless an empirical isotonic regression or Platt scaling calibration artifact has been fitted and validated on the target clinic camera distribution, the system never claims calibrated uncertainty.

### 4.2 Decision Margin & Runner-Up Identification
To provide screeners with visibility into borderline classifications:
- **Top Probability**: $P_{\text{top}} = \max_c P(c)$
- **Runner-Up Probability**: $P_{\text{runner-up}} = \max_{c \ne \text{top}} P(c)$
- **Decision Margin**: $\Delta P = P_{\text{top}} - P_{\text{runner-up}}$

If $\Delta P < 15.0\%$, the workstation explicitly highlights the borderline margin, warning the screener that the classification between the top grade and runner-up grade was contested.

---

## 5. Referral Logic: Prototype Screening Rule

RetinaGuard 2.0 enforces a configured prototype screening rule:
$$\text{Referral Flag} = \begin{cases} \text{REFERABLE DR}, & \text{if } \text{Predicted Grade} \ge 2 \\ \text{NON-REFERABLE}, & \text{if } \text{Predicted Grade} < 2 \end{cases}$$

### Explicit Regulatory Disclaimer:
This threshold is a **configured prototype screening rule** based on standard epidemiological screening conventions (ICDR Moderate NPDR and above typically warrant specialist referral). It does **not** constitute an independently validated clinical guideline or autonomous regulatory-cleared diagnostic criteria.

---

## 6. Intended Use & Boundaries

### 6.1 Intended Use:
- **Purpose**: AI-assisted diabetic retinopathy screening decision-support workstation for trained healthcare workers in rural or community clinics.
- **Input**: Digital color retinal fundus camera images (Macula-centered or Disc-centered, 45-degree field of view).
- **Output**: Screening triage recommendation, candidate lesion evidence, decision margin, and Grad-CAM++ attribution map for ophthalmic review.

### 6.2 Out-of-Scope & Contraindications:
- **Autonomous Diagnosis**: Never use as a standalone autonomous diagnostic device without human physician oversight.
- **Non-DR Pathologies**: Not validated for autonomous diagnosis of glaucoma, age-related macular degeneration (AMD), or retinal vein occlusion.
- **Pediatric Patients**: Not validated on pediatric populations (retinopathy of prematurity).
- **Ungradable Scans**: Any scan flagged as UNGRADABLE must be recaptured; downstream predictions must never be forced or simulated.
