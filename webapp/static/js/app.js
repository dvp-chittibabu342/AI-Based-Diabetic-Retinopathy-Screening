/**
 * RetinaGuard 2.0 Web Application Controller
 * Premium AI-Assisted Retinal Screening Workstation Logic
 * Team VyuhaX • SIH 2026 PS: SIH26038
 *
 * Architectural Principles:
 * 1. Fixed Left Analysis Canvas / Scrolling Right Information Workspace
 * 2. High-Contrast WCAG-Compliant Pure #FFFFFF / #000000 / Deep Red #D71920 (Zero Blue)
 * 3. Evidence <-> Image <-> Explanation Interaction Pattern
 * 4. Grounded ExplainAI in 8 Mandated Sections (01-08)
 * 5. Focus Evidence Mode (Expand Canvas, Minimize Right Column, ESC returns)
 * 6. Compact Status Rail (QUALITY ✓ • EVIDENCE ✓ • MODEL ✓ • XAI ✓)
 * 7. Real Runtime Execution Latencies (Zero Fabricated Timings)
 * 8. Strict Parity with Unified AnalysisResult Contract
 */

document.addEventListener("DOMContentLoaded", () => {
  // =========================================================================
  // State
  // =========================================================================
  let currentFile = null;
  let currentDemoId = null;
  let currentAnalysis = null;
  let currentOverlays = {};
  let demoCases = [];
  let currentRequestId = 0;
  let isFocusMode = false;

  // =========================================================================
  // DOM Selectors
  // =========================================================================
  // Header & Controls
  const selectDemoCase = document.getElementById("select-demo-case");
  const btnLoadDemo = document.getElementById("btn-load-demo");
  const btnTriggerUpload = document.getElementById("btn-trigger-upload");
  const fileUpload = document.getElementById("file-upload");
  const btnAnalyze = document.getElementById("btn-analyze");
  const analyzeSpinner = document.getElementById("analyze-spinner");
  const analyzeText = document.getElementById("analyze-text");
  
  // Patient Details & Metadata
  const inputPatientId = document.getElementById("input-patient-id");
  const inputExamId = document.getElementById("input-exam-id");
  const selectEye = document.getElementById("select-eye");
  const valMetaCase = document.getElementById("val-meta-case");
  const valMetaEye = document.getElementById("val-meta-eye");
  const imageMetaTag = document.getElementById("image-meta-tag");

  // Canvas Stage & Viewport
  const workstationShell = document.getElementById("workstation-shell");
  const btnFocusMode = document.getElementById("btn-focus-mode");
  const emptyPlaceholder = document.getElementById("empty-placeholder");
  const mainImageView = document.getElementById("main-image-view");
  const evidenceSvgOverlay = document.getElementById("evidence-svg-overlay");
  const analysisScanLine = document.getElementById("analysis-scan-line");
  const retinaFocusRing = document.getElementById("retina-focus-ring");
  const processingOverlay = document.getElementById("processing-overlay");
  
  // Canvas Layer & Active Evidence
  const layerToggles = document.getElementById("layer-toggles");
  const layerLegend = document.getElementById("layer-legend");
  const safetyAlertBox = document.getElementById("safety-alert-box");
  const borderlineAlertBox = document.getElementById("borderline-alert-box");
  const attributionTargetSelector = document.getElementById("attribution-target-selector");
  const selectTargetClass = document.getElementById("select-target-class");
  const activeEvidenceIndicator = document.getElementById("active-evidence-indicator");
  const activeEvidenceTitle = document.getElementById("active-evidence-title");
  const activeEvidenceCoords = document.getElementById("active-evidence-coords");
  const btnClearEvidence = document.getElementById("btn-clear-evidence");

  // Candidate Crops & Region Sensitivity
  const evidenceCard = document.getElementById("evidence-card");
  const badgeCropCount = document.getElementById("badge-crop-count");
  const candidateCropsContainer = document.getElementById("candidate-crops-container");
  const btnRunSensitivity = document.getElementById("btn-run-sensitivity");
  const sensitivityResultsWrap = document.getElementById("sensitivity-results-wrap");

  // Right Column: Status Rail
  const railQuality = document.getElementById("rail-quality");
  const railEvidence = document.getElementById("rail-evidence");
  const railModel = document.getElementById("rail-model");
  const railXai = document.getElementById("rail-xai");

  // Right Column: Decision Core (Result)
  const badgeReferralStatus = document.getElementById("badge-referral-status");
  const textGradeHeadline = document.getElementById("text-grade-headline");
  const textGradeConfidence = document.getElementById("text-grade-confidence");
  const textCalibrationStatus = document.getElementById("text-calibration-status");
  const marginIndicatorCard = document.getElementById("margin-indicator-card");
  const valDecisionMargin = document.getElementById("val-decision-margin");
  const demoComparisonBox = document.getElementById("demo-comparison-box");
  const valReferenceGrade = document.getElementById("val-reference-grade");
  const valAiPrediction = document.getElementById("val-ai-prediction");
  const probabilitiesSection = document.getElementById("probabilities-section");
  const probBarsContainer = document.getElementById("prob-bars-container");
  const recommendationBox = document.getElementById("recommendation-box");
  const recTitle = document.getElementById("rec-title");
  const recBody = document.getElementById("rec-body");
  const recRule = document.getElementById("rec-rule");
  const btnOpenExplain = document.getElementById("btn-open-explain");
  const reportActionBar = document.getElementById("report-action-bar");
  const btnViewReport = document.getElementById("btn-view-report");
  const btnDownloadReport = document.getElementById("btn-download-report");

  // Right Column: Inspect Buttons & Expandable Panels
  const btnInspectQuality = document.getElementById("btn-inspect-quality");
  const panelQualityDetails = document.getElementById("panel-quality-details");
  const badgeQualityStatus = document.getElementById("badge-quality-status");
  const valQualityScore = document.getElementById("val-quality-score");
  const qualityMeterFill = document.getElementById("quality-meter-fill");
  const valQFocus = document.getElementById("val-q-focus");
  const valQIllum = document.getElementById("val-q-illum");
  const valQContrast = document.getElementById("val-q-contrast");
  const valQFov = document.getElementById("val-q-fov");
  const boxQualityReason = document.getElementById("box-quality-reason");
  const textQualityReason = document.getElementById("text-quality-reason");

  const btnInspectEvidence = document.getElementById("btn-inspect-evidence");
  const panelEvidenceDetails = document.getElementById("panel-evidence-details");
  const badgeLesionMode = document.getElementById("badge-lesion-mode");
  const valMaCount = document.getElementById("val-ma-count");
  const valMaNotes = document.getElementById("val-ma-notes");
  const valExCount = document.getElementById("val-ex-count");
  const valExNotes = document.getElementById("val-ex-notes");
  const valHeCount = document.getElementById("val-he-count");
  const valHeNotes = document.getElementById("val-he-notes");
  const valSeCount = document.getElementById("val-se-count");
  const valSeNotes = document.getElementById("val-se-notes");
  const valVesselDensity = document.getElementById("val-vessel-density");
  const valNvRisk = document.getElementById("val-nv-risk");
  const valOdStatus = document.getElementById("val-od-status");
  const valOdDetails = document.getElementById("val-od-details");
  const valFoveaStatus = document.getElementById("val-fovea-status");
  const valFoveaDetails = document.getElementById("val-fovea-details");

  const btnInspectTrace = document.getElementById("btn-inspect-trace");
  const panelTraceDetails = document.getElementById("panel-trace-details");
  const valTraceTotal = document.getElementById("val-trace-total");
  const traceQuality = document.getElementById("trace-quality");
  const traceEnhancement = document.getElementById("trace-enhancement");
  const traceAnatomy = document.getElementById("trace-anatomy");
  const traceEvidence = document.getElementById("trace-evidence");
  const traceGrading = document.getElementById("trace-grading");
  const traceAttribution = document.getElementById("trace-attribution");
  const traceReport = document.getElementById("trace-report");

  // ExplainAI Drawer
  const explainaiDrawerOverlay = document.getElementById("explainai-drawer-overlay");
  const btnCloseExplain = document.getElementById("btn-close-explain");
  const explainaiQaList = document.getElementById("explainai-qa-list");

  // Transparency Stage Modal
  const stageModal = document.getElementById("stage-modal");
  const btnCloseStage = document.getElementById("btn-close-stage");
  const modalStageCategory = document.getElementById("modal-stage-category");
  const modalStageName = document.getElementById("modal-stage-name");
  const modalStageWhat = document.getElementById("modal-stage-what");
  const modalStageWhy = document.getElementById("modal-stage-why");
  const modalStageParams = document.getElementById("modal-stage-params");
  const modalStageChanged = document.getElementById("modal-stage-changed");
  const modalStageLimitation = document.getElementById("modal-stage-limitation");
  const btnStages = document.getElementById("btn-stages");

  // Validation & About Modals
  const validationModal = document.getElementById("validation-modal");
  const btnCloseValidation = document.getElementById("btn-close-validation");
  const btnValidation = document.getElementById("btn-validation");
  const aboutModal = document.getElementById("about-modal");
  const btnCloseAbout = document.getElementById("btn-close-about");
  const btnAbout = document.getElementById("btn-about");

  // =========================================================================
  // Metadata Sync
  // =========================================================================
  function updateMetadataChips(statusText = "IDLE") {
    const examId = inputExamId.value || "EX-00412";
    const eyeVal = selectEye.value.includes("Left") ? "OS" : "OD";
    if (valMetaCase) valMetaCase.textContent = examId;
    if (valMetaEye) valMetaEye.textContent = eyeVal;
    if (imageMetaTag) imageMetaTag.innerHTML = `STATUS: <strong>${statusText}</strong>`;
  }

  if (inputExamId) inputExamId.addEventListener("input", () => updateMetadataChips("IDLE"));
  if (selectEye) selectEye.addEventListener("change", () => updateMetadataChips("IDLE"));

  // =========================================================================
  // 1. Fetch and Populate Authentic Reference Demo Cases
  // =========================================================================
  async function loadDemoCases() {
    try {
      const resp = await fetch("/api/demo-cases?all=true");
      if (!resp.ok) return;
      demoCases = await resp.json();
      if (!demoCases || demoCases.length === 0) {
        selectDemoCase.innerHTML = '<option value="">-- No demo scans bundled (Upload below) --</option>';
        selectDemoCase.disabled = true;
        btnLoadDemo.disabled = true;
      } else {
        selectDemoCase.innerHTML = '<option value="">Select Reference Case (1–6)</option>';
        demoCases.forEach(c => {
          const opt = document.createElement("option");
          opt.value = c.id;
          opt.textContent = c.label;
          selectDemoCase.appendChild(opt);
        });
        selectDemoCase.disabled = false;
        btnLoadDemo.disabled = false;
      }
    } catch (err) {
      console.error("Failed to load demo cases:", err);
    }
  }
  loadDemoCases();

  function handleSelectDemoCase(demoId) {
    if (!demoId) {
      clearSelection();
      return;
    }

    const selectedCase = demoCases.find(c => c.id === demoId);
    if (!selectedCase) return;

    currentDemoId = demoId;
    currentFile = null;
    fileUpload.value = "";
    selectDemoCase.value = demoId;
    currentRequestId++;

    showImage(selectedCase.image_url, selectedCase.label);
    updateMetadataChips("READY");

    resetResults(`Loaded authentic reference case (${selectedCase.reference_name}). Click Analyze Scan ▶.`);
    btnAnalyze.disabled = false;
  }

  function clearSelection() {
    currentDemoId = null;
    currentFile = null;
    fileUpload.value = "";
    selectDemoCase.value = "";
    currentRequestId++;

    emptyPlaceholder.style.display = "flex";
    mainImageView.style.display = "none";
    mainImageView.src = "";
    updateMetadataChips("IDLE");

    resetResults("Select a demo case or upload a retinal scan to begin screening.");
    btnAnalyze.disabled = true;
  }

  selectDemoCase.addEventListener("change", (e) => handleSelectDemoCase(e.target.value));
  btnLoadDemo.addEventListener("click", () => handleSelectDemoCase(selectDemoCase.value));

  // =========================================================================
  // 2. Upload Handling
  // =========================================================================
  btnTriggerUpload.addEventListener("click", () => fileUpload.click());

  fileUpload.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;

    currentFile = file;
    currentDemoId = null;
    selectDemoCase.value = "";
    currentRequestId++;

    const objectUrl = URL.createObjectURL(file);
    showImage(objectUrl, file.name);

    updateMetadataChips("READY");
    resetResults("Image loaded from scan file. Click Analyze Scan ▶ to run screening.");
    btnAnalyze.disabled = false;
  });

  function showImage(src, alt) {
    emptyPlaceholder.style.display = "none";
    mainImageView.style.display = "block";
    mainImageView.src = src;
    mainImageView.alt = alt;
    mainImageView.style.opacity = "1";
    clearEvidenceHighlight();
  }

  // =========================================================================
  // 3. Reset Results UI
  // =========================================================================
  function resetResults(msg) {
    currentAnalysis = null;
    currentOverlays = {};
    clearEvidenceHighlight();

    // Hide overlays & alert banners
    layerToggles.style.display = "none";
    document.querySelectorAll(".layer-tab-group .toggle-btn").forEach(b => {
      b.classList.toggle("active", b.getAttribute("data-layer") === "original");
    });
    if (layerLegend) {
      layerLegend.style.display = "none";
      layerLegend.innerHTML = "";
    }
    if (attributionTargetSelector) {
      attributionTargetSelector.style.display = "none";
    }
    if (selectTargetClass) {
      selectTargetClass.value = "predicted";
    }
    safetyAlertBox.style.display = "none";
    safetyAlertBox.innerHTML = "";
    borderlineAlertBox.style.display = "none";
    borderlineAlertBox.innerHTML = "";
    probabilitiesSection.style.display = "none";
    probBarsContainer.innerHTML = "";
    reportActionBar.style.display = "none";

    // Status Rail Reset
    [railQuality, railEvidence, railModel, railXai].forEach(r => {
      if (r) {
        r.classList.remove("ready", "failed");
        const chk = r.querySelector(".rail-check");
        if (chk) chk.textContent = "✓";
      }
    });

    // Evidence Gallery & Sensitivity
    if (evidenceCard) evidenceCard.style.display = "none";
    if (candidateCropsContainer) candidateCropsContainer.innerHTML = "";
    if (sensitivityResultsWrap) {
      sensitivityResultsWrap.style.display = "none";
      sensitivityResultsWrap.innerHTML = "";
    }
    if (btnRunSensitivity) {
      btnRunSensitivity.disabled = false;
      btnRunSensitivity.textContent = "Test Sensitivity";
    }

    // ICDR Scale & Margin
    document.querySelectorAll(".icdr-step").forEach(step => step.classList.remove("active-grade"));
    if (marginIndicatorCard) marginIndicatorCard.style.display = "none";
    if (textCalibrationStatus) {
      textCalibrationStatus.textContent = "Calibration: not applied (raw model probabilities)";
    }

    // Quality Panel
    badgeQualityStatus.textContent = "PENDING";
    badgeQualityStatus.className = "chip chip-neutral";
    valQualityScore.textContent = "--";
    qualityMeterFill.style.width = "0%";
    qualityMeterFill.style.background = "#000000";
    valQFocus.textContent = "--";
    valQIllum.textContent = "--";
    valQContrast.textContent = "--";
    valQFov.textContent = "--";
    panelQualityDetails.style.display = "none";
    if (btnInspectQuality) btnInspectQuality.textContent = "Quality Details →";
    boxQualityReason.style.display = "none";
    textQualityReason.textContent = "--";

    // DR Severity Panel
    badgeReferralStatus.textContent = "PENDING";
    badgeReferralStatus.className = "chip chip-neutral";
    textGradeHeadline.textContent = "Awaiting Analysis";
    textGradeHeadline.style.color = "#000000";
    textGradeConfidence.textContent = msg || "Load a scan and execute screening";
    if (demoComparisonBox) demoComparisonBox.style.display = "none";
    if (valReferenceGrade) valReferenceGrade.textContent = "--";
    if (valAiPrediction) valAiPrediction.textContent = "--";

    // Candidate Evidence & Landmarks
    badgeLesionMode.textContent = "PENDING";
    badgeLesionMode.className = "chip chip-neutral";
    panelEvidenceDetails.style.display = "none";
    if (btnInspectEvidence) btnInspectEvidence.textContent = "Evidence Details →";
    valOdStatus.textContent = "--";
    if (valOdDetails) valOdDetails.textContent = "(Masked)";
    valFoveaStatus.textContent = "--";
    if (valFoveaDetails) valFoveaDetails.textContent = "(Projection)";
    valVesselDensity.textContent = "--";
    valMaCount.textContent = "--";
    if (valMaNotes) valMaNotes.textContent = "Candidate findings";
    valExCount.textContent = "--";
    if (valExNotes) valExNotes.textContent = "Candidate findings";
    valHeCount.textContent = "--";
    if (valHeNotes) valHeNotes.textContent = "Candidate findings";
    if (valSeCount) valSeCount.textContent = "--";
    if (valSeNotes) valSeNotes.textContent = "Cotton wool spots";
    valNvRisk.textContent = "--";

    // Trace Panel
    valTraceTotal.textContent = "-- ms";
    panelTraceDetails.style.display = "none";
    if (btnInspectTrace) btnInspectTrace.textContent = "Trace Details →";
    traceQuality.textContent = "-- ms";
    traceEnhancement.textContent = "-- ms";
    traceAnatomy.textContent = "-- ms";
    traceEvidence.textContent = "-- ms";
    traceGrading.textContent = "-- ms";
    traceAttribution.textContent = "-- ms";
    traceReport.textContent = "-- ms";

    // ExplainAI Panel
    if (explainaiQaList) {
      explainaiQaList.innerHTML = '<div class="empty-state-text">Run screening analysis to generate evidence-grounded clinical explanations.</div>';
    }

    // Recommendation & Report
    recTitle.textContent = "Standard Triage Recommendation";
    recBody.textContent = msg || "Upload or select a case to generate AI-assisted screening interpretation.";
    if (btnViewReport) btnViewReport.removeAttribute("href");
    if (btnDownloadReport) {
      btnDownloadReport.removeAttribute("href");
      btnDownloadReport.removeAttribute("download");
    }

    // Stepper reset
    document.querySelectorAll(".step-node").forEach(node => node.classList.remove("active-stage"));
  }

  // =========================================================================
  // 4. Run Analysis
  // =========================================================================
  btnAnalyze.addEventListener("click", () => executeAnalysis());

  async function executeAnalysis(targetClassOverride = null) {
    if (!currentFile && !currentDemoId) return;

    btnAnalyze.disabled = true;
    analyzeSpinner.style.display = "inline-block";
    analyzeText.textContent = "Analyzing Scan...";
    processingOverlay.style.display = "flex";
    
    // Scan line animation & Focus Ring pulse
    if (analysisScanLine) analysisScanLine.style.display = "block";
    if (retinaFocusRing) retinaFocusRing.classList.add("analyzing");
    updateMetadataChips("ANALYZING...");

    const requestId = ++currentRequestId;

    if (!targetClassOverride) {
      resetResults("Executing screening pipeline...");
    }

    const formData = new FormData();
    if (currentFile) {
      formData.append("file", currentFile);
    } else if (currentDemoId) {
      formData.append("demo_id", currentDemoId);
    }

    formData.append("patient_id", inputPatientId.value || "PT-82910");
    formData.append("exam_id", inputExamId.value || "EX-00412");
    formData.append("eye", selectEye.value || "Right (OD)");

    if (targetClassOverride !== null && targetClassOverride !== undefined && targetClassOverride !== "predicted") {
      formData.append("target_class", targetClassOverride);
    }

    try {
      const resp = await fetch("/api/analyze", {
        method: "POST",
        body: formData
      });

      if (!resp.ok) {
        const errorData = await resp.json();
        throw new Error(errorData.detail || "Screening analysis failed.");
      }

      const result = await resp.json();

      if (requestId !== currentRequestId) {
        console.warn("Discarding stale analysis response for older request", requestId);
        return;
      }

      currentAnalysis = result;
      renderAnalysisResults(result);
    } catch (err) {
      if (requestId === currentRequestId) {
        alert("Screening Error: " + err.message);
        resetResults("Analysis error occurred: " + err.message);
      }
    } finally {
      if (requestId === currentRequestId) {
        btnAnalyze.disabled = false;
        analyzeSpinner.style.display = "none";
        analyzeText.textContent = "Analyze Scan ▶";
        processingOverlay.style.display = "none";
        if (analysisScanLine) analysisScanLine.style.display = "none";
        if (retinaFocusRing) retinaFocusRing.classList.remove("analyzing");
      }
    }
  }

  // Dynamic Target Class Selection Change
  if (selectTargetClass) {
    selectTargetClass.addEventListener("change", (e) => {
      const targetVal = e.target.value;
      if (currentAnalysis && !currentAnalysis.safety_gate_triggered) {
        executeAnalysis(targetVal);
      }
    });
  }

  // =========================================================================
  // 5. Render Full Results (AnalysisResult Contract)
  // =========================================================================
  function renderAnalysisResults(data) {
    const q = data.quality;
    const isUngradable = data.safety_gate_triggered || !q.gradable;

    updateMetadataChips(isUngradable ? "UNGRADABLE" : "GRADABLE");

    // Reset banners
    safetyAlertBox.style.display = "none";
    safetyAlertBox.innerHTML = "";
    borderlineAlertBox.style.display = "none";
    borderlineAlertBox.innerHTML = "";

    // Stepper node highlighting
    document.querySelectorAll(".step-node").forEach(node => node.classList.add("active-stage"));

    // Compact Status Rail: QUALITY
    if (railQuality) {
      railQuality.classList.add(isUngradable ? "failed" : "ready");
      const chk = railQuality.querySelector(".rail-check");
      if (chk) chk.textContent = isUngradable ? "✕" : "✓";
    }

    // Quality Panel
    valQualityScore.textContent = q.score;
    qualityMeterFill.style.width = `${Math.min(100, Math.max(0, q.score))}%`;
    valQFocus.textContent = q.focus;
    valQIllum.textContent = q.illumination;
    valQContrast.textContent = q.contrast;
    valQFov.textContent = `${q.fov_percent}%`;

    badgeQualityStatus.textContent = q.status;
    if (q.status === "GOOD") {
      badgeQualityStatus.className = "chip chip-good";
      qualityMeterFill.style.background = "#15803d";
    } else if (q.status === "BORDERLINE") {
      badgeQualityStatus.className = "chip chip-warning";
      qualityMeterFill.style.background = "#b45309";
    } else {
      badgeQualityStatus.className = "chip chip-danger";
      qualityMeterFill.style.background = "#b91c1c";
    }

    boxQualityReason.style.display = "block";
    textQualityReason.textContent = q.reason;

    // Execution Trace Timings (Real runtime from AnalysisResult)
    const t = data.timings || {};
    valTraceTotal.textContent = t.total_ms !== undefined ? `${t.total_ms} ms` : "-- ms";
    traceQuality.textContent = t.quality_ms !== undefined ? `${t.quality_ms} ms` : "-- ms";
    traceEnhancement.textContent = t.enhancement_ms !== undefined ? `${t.enhancement_ms} ms` : "-- ms";
    traceAnatomy.textContent = t.anatomy_ms !== undefined ? `${t.anatomy_ms} ms` : "-- ms";
    traceEvidence.textContent = t.lesions_ms !== undefined ? `${t.lesions_ms} ms` : "-- ms";
    traceGrading.textContent = t.grading_ms !== undefined ? `${t.grading_ms} ms` : "-- ms";
    if (t.attribution_ms !== undefined) {
      traceAttribution.textContent = `${t.attribution_ms} ms`;
    } else if (data.xai && data.xai.status === "SUCCESS") {
      traceAttribution.textContent = `Included in model pass (${data.xai.method || "Grad-CAM++"})`;
    } else {
      traceAttribution.textContent = "N/A";
    }
    traceReport.textContent = t.report_ms !== undefined ? `${t.report_ms} ms` : "-- ms";

    // -----------------------------------------------------------------------
    // CRITICAL SAFETY GATE: UNGRADABLE SCAN
    // -----------------------------------------------------------------------
    if (isUngradable) {
      safetyAlertBox.style.display = "block";
      safetyAlertBox.innerHTML = `
        <strong>IMAGE UNGRADABLE:</strong> Downstream grading stopped for patient safety.<br>
        <em>${q.reason}</em><br>
        <strong>Recommended Action:</strong> ${q.recommendation}
      `;

      badgeReferralStatus.textContent = "UNGRADABLE";
      badgeReferralStatus.className = "chip chip-danger";
      textGradeHeadline.textContent = "Screening Halted";
      textGradeHeadline.style.color = "#b91c1c";

      const ref = data.demo_reference;
      if (ref) {
        textGradeConfidence.innerHTML = `<strong>Reference:</strong> ${ref.label} &bull; <strong>Status:</strong> Rejected by Optical Quality Gate (Score: ${q.score}/100)`;
      } else {
        textGradeConfidence.textContent = "Optical quality is insufficient to produce a reliable screening prediction.";
      }

      // Hide probabilities
      probabilitiesSection.style.display = "none";
      probBarsContainer.innerHTML = "";
      if (marginIndicatorCard) marginIndicatorCard.style.display = "none";

      // Stepper: highlight stage 1, unhighlight downstream
      document.querySelectorAll(".step-node").forEach(node => {
        if (node.getAttribute("data-stage") !== "stage_1_quality_gate") {
          node.classList.remove("active-stage");
        }
      });

      // Update remaining status rails as halted
      [railEvidence, railModel, railXai].forEach(r => {
        if (r) {
          r.classList.add("failed");
          const chk = r.querySelector(".rail-check");
          if (chk) chk.textContent = "✕";
        }
      });

      badgeLesionMode.textContent = "HALTED";
      badgeLesionMode.className = "chip chip-danger";
      valOdStatus.textContent = "Halted";
      if (valOdDetails) valOdDetails.textContent = "Downstream analysis withheld";
      valFoveaStatus.textContent = "Halted";
      if (valFoveaDetails) valFoveaDetails.textContent = "Downstream analysis withheld";
      valVesselDensity.textContent = "N/A";
      valMaCount.textContent = "N/A";
      if (valMaNotes) valMaNotes.textContent = "Safety Gate Triggered";
      valExCount.textContent = "N/A";
      if (valExNotes) valExNotes.textContent = "Safety Gate Triggered";
      valHeCount.textContent = "N/A";
      if (valHeNotes) valHeNotes.textContent = "Safety Gate Triggered";
      if (valSeCount) valSeCount.textContent = "N/A";
      if (valSeNotes) valSeNotes.textContent = "Safety Gate Triggered";
      valNvRisk.textContent = "N/A";

      recTitle.textContent = "Safety Rejection Guidance";
      recBody.textContent = q.recommendation;

      layerToggles.style.display = "none";
      if (layerLegend) layerLegend.style.display = "none";
      currentOverlays = {};

      renderExplainAI(data.explanation);

      if (data.report) {
        setupReportLinks(data.report);
      }
      return;
    }

    // -----------------------------------------------------------------------
    // GRADABLE SCAN: POPULATE ALL WORKSTATION PANELS
    // -----------------------------------------------------------------------
    // Borderline Alert
    if (q.status === "BORDERLINE") {
      borderlineAlertBox.style.display = "block";
      borderlineAlertBox.innerHTML = `
        <strong>BORDERLINE QUALITY:</strong> Automated green-channel CLAHE enhancement was applied.<br>
        <em>${q.reason}</em>
      `;
    }

    // Landmarks & Evidence Analysis
    const st = data.structures || data.anatomy || {};
    const l = data.lesions || data.lesion_candidates || {};

    if (railEvidence) {
      railEvidence.classList.add("ready");
      const chk = railEvidence.querySelector(".rail-check");
      if (chk) chk.textContent = "✓";
    }

    valOdStatus.textContent = st.optic_disc?.centroid ? `Centroid [${st.optic_disc.centroid.join(", ")}]` : "--";
    if (valOdDetails) valOdDetails.textContent = st.optic_disc?.radius_px ? `Radius: ${st.optic_disc.radius_px} px (${st.optic_disc.methodology || "AI/Morphological Mask"})` : "(Masked)";
    valFoveaStatus.textContent = st.fovea?.coordinates ? `Coords [${st.fovea.coordinates.join(", ")}]` : "--";
    if (valFoveaDetails) valFoveaDetails.textContent = st.fovea?.methodology || "Temporal Geometric Projection";
    valVesselDensity.textContent = st.vessels?.density_percent !== undefined ? `${st.vessels.density_percent}%` : "--";

    if (badgeLesionMode) {
      if (l.is_ai || l.mode === "AI SEGMENTATION") {
        badgeLesionMode.textContent = "CANDIDATE EVIDENCE";
        badgeLesionMode.className = "chip chip-good";
      } else {
        badgeLesionMode.textContent = "HEURISTIC CANDIDATES";
        badgeLesionMode.className = "chip chip-neutral";
      }
    }

    valMaCount.textContent = `${l.microaneurysms?.candidate_count || 0} candidates`;
    if (valMaNotes) valMaNotes.textContent = l.is_ai ? "U-Net Segmented Candidates" : "Morphological Top-Hat";

    valExCount.textContent = `${l.hard_exudates?.candidate_count || 0} candidates (${l.hard_exudates?.total_area_px || 0} px)`;
    if (valExNotes) valExNotes.textContent = l.is_ai ? "U-Net Segmented Candidates" : "Luminance / Disc-Masked";

    valHeCount.textContent = `${l.hemorrhages?.candidate_count || 0} candidates (${l.hemorrhages?.total_area_px || 0} px)`;
    if (valHeNotes) valHeNotes.textContent = l.is_ai ? "U-Net Segmented Candidates" : "Dark Intra-retinal Candidates";

    if (valSeCount) {
      valSeCount.textContent = l.soft_exudates ? `${l.soft_exudates.candidate_count || 0} candidates (${l.soft_exudates.total_area_px || 0} px)` : "0 candidates";
    }
    if (valSeNotes) valSeNotes.textContent = "Cotton Wool Spot Candidates";

    valNvRisk.textContent = l.neovascularization?.risk_level ? `${l.neovascularization.risk_level} (${l.neovascularization.density_heuristic || "Density Heuristic"})` : "--";

    // Candidate Evidence Crops Gallery (Left Workspace)
    renderCandidateCrops(l.visual_crops || l.candidate_crops || []);

    // DR Grading (EXP-001)
    const g = data.grading || data.model || {};
    if (railModel) {
      railModel.classList.add("ready");
      const chk = railModel.querySelector(".rail-check");
      if (chk) chk.textContent = "✓";
    }

    textGradeHeadline.textContent = `Grade ${g.grade} — ${g.severity_name}`;
    textGradeHeadline.style.color = getGradeColor(g.grade);

    // Update 5-stage ICDR scale progression marker
    document.querySelectorAll(".icdr-step").forEach(step => {
      const stepGrade = parseInt(step.getAttribute("data-grade"), 10);
      step.classList.toggle("active-grade", stepGrade === g.grade);
    });

    // Top-2 Decision Margin
    if (marginIndicatorCard && g.decision_margin_percent !== undefined) {
      marginIndicatorCard.style.display = "flex";
      valDecisionMargin.textContent = `${g.decision_margin_percent}% margin over runner-up (${g.runner_up_name || 'Grade ' + g.runner_up_grade})`;
    }

    if (textCalibrationStatus) {
      textCalibrationStatus.textContent = g.calibration_status || "Calibration: not applied (raw model probabilities)";
    }

    // Authentic Reference vs Model Comparison
    const ref = data.demo_reference;
    if (ref && ref.reference_grade !== undefined && ref.reference_grade >= 0) {
      if (demoComparisonBox) demoComparisonBox.style.display = "grid";
      if (valReferenceGrade) valReferenceGrade.textContent = `Grade ${ref.reference_grade} — ${ref.reference_name}`;
      if (valAiPrediction) valAiPrediction.textContent = `Grade ${g.grade} — ${g.severity_name}`;
      textGradeConfidence.textContent = `Model probability: ${g.confidence_percent}% (${g.confidence_level}) • Model: EXP-001`;
    } else {
      if (demoComparisonBox) demoComparisonBox.style.display = "none";
      textGradeConfidence.textContent = `Model probability: ${g.confidence_percent}% (${g.confidence_level}) • Model: EXP-001`;
    }

    if (g.referral) {
      badgeReferralStatus.textContent = "REFERABLE DR (YES)";
      badgeReferralStatus.className = "chip chip-danger";
    } else {
      badgeReferralStatus.textContent = "NON-REFERABLE (NO)";
      badgeReferralStatus.className = "chip chip-good";
    }

    // Full 5-Class Probabilities
    probBarsContainer.innerHTML = "";
    (g.probabilities || []).forEach(p => {
      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = `
        <span class="prob-label">Grade ${p.grade} (${p.label})</span>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" style="width: ${p.percentage}%; background: ${p.grade === g.grade ? getGradeColor(p.grade) : '#d1d5db'};"></div>
        </div>
        <span class="prob-pct">${p.percentage}%</span>
      `;
      probBarsContainer.appendChild(row);
    });
    probabilitiesSection.style.display = "block";

    // Recommendation & Screening Interpretation
    recTitle.textContent = g.referral_action || "Standard Triage";
    recBody.textContent = g.recommendation;
    if (recRule) recRule.textContent = g.referral_rule || "Configured prototype screening rule: Grade ≥ 2 triggers referable triage.";

    // XAI Attribution Status
    if (railXai) {
      if (data.xai && data.xai.status === "SUCCESS") {
        railXai.classList.add("ready");
        const chk = railXai.querySelector(".rail-check");
        if (chk) chk.textContent = "✓";
      } else {
        railXai.classList.add("failed");
        const chk = railXai.querySelector(".rail-check");
        if (chk) chk.textContent = "✕";
      }
    }

    // Overlays & Layers
    currentOverlays = data.overlays || {};
    layerToggles.style.display = "flex";
    setActiveLayer("original");

    // ExplainAI Drawer Population
    renderExplainAI(data.explanation);

    // Report Links
    if (data.report) {
      setupReportLinks(data.report);
    }
  }

  function getGradeColor(grade) {
    const colors = ["#15803d", "#000000", "#b45309", "#c2410c", "#b91c1c"];
    return colors[grade] || "#000000";
  }

  // =========================================================================
  // 6. Candidate Evidence Crops Gallery & Evidence ↔ Image Interaction
  // =========================================================================
  function renderCandidateCrops(crops) {
    if (!candidateCropsContainer) return;

    if (!crops || crops.length === 0) {
      if (evidenceCard) evidenceCard.style.display = "none";
      candidateCropsContainer.innerHTML = "";
      if (badgeCropCount) badgeCropCount.textContent = "0 Candidates";
      return;
    }

    if (evidenceCard) evidenceCard.style.display = "block";
    if (badgeCropCount) badgeCropCount.textContent = `${crops.length} Candidates`;
    candidateCropsContainer.innerHTML = "";

    crops.forEach(c => {
      const imgSrc = c.crop_base64 ? `data:image/jpeg;base64,${c.crop_base64}` : (c.crop_url || c.preview_url || "");
      const cType = c.candidate_type || c.type || "Candidate";
      const bbox = c.bounding_box || { x: (c.bbox ? c.bbox[0] : 0), y: (c.bbox ? c.bbox[1] : 0), width: (c.bbox ? c.bbox[2] : 96), height: (c.bbox ? c.bbox[3] : 96) };

      const card = document.createElement("div");
      card.className = "crop-card";
      card.setAttribute("title", c.description || cType);
      card.innerHTML = `
        <img src="${imgSrc}" class="crop-img" alt="${cType}">
        <div class="crop-label">${cType}</div>
        <span class="crop-badge">${bbox.width}×${bbox.height} px</span>
      `;

      card.addEventListener("click", () => {
        selectEvidenceRegion(c, bbox, card);
      });

      candidateCropsContainer.appendChild(card);
    });
  }

  function selectEvidenceRegion(c, bbox, card) {
    // 1. Highlight card in gallery
    document.querySelectorAll(".crop-card").forEach(el => el.classList.remove("active-crop"));
    if (card) card.classList.add("active-crop");

    const cType = c.candidate_type || c.type || "Candidate Finding";
    const x = bbox.x !== undefined ? bbox.x : (c.bbox ? c.bbox[0] : 0);
    const y = bbox.y !== undefined ? bbox.y : (c.bbox ? c.bbox[1] : 0);
    const w = bbox.width !== undefined ? bbox.width : (c.bbox ? c.bbox[2] : 96);
    const h = bbox.height !== undefined ? bbox.height : (c.bbox ? c.bbox[3] : 96);

    // 2. Update Canvas Subbar Indicator
    if (activeEvidenceIndicator) {
      activeEvidenceIndicator.style.display = "flex";
      if (activeEvidenceTitle) activeEvidenceTitle.textContent = cType;
      if (activeEvidenceCoords) activeEvidenceCoords.textContent = `[Coord: X: ${x}, Y: ${y} • Size: ${w}×${h} px]`;
    }

    // 3. Draw SVG Overlay on retinal image
    if (evidenceSvgOverlay && mainImageView) {
      const natW = mainImageView.naturalWidth || 512;
      const natH = mainImageView.naturalHeight || 512;
      evidenceSvgOverlay.setAttribute("viewBox", `0 0 ${natW} ${natH}`);
      evidenceSvgOverlay.style.display = "block";

      const tagWidth = Math.max(90, cType.length * 8);

      evidenceSvgOverlay.innerHTML = `
        <defs>
          <filter id="evidenceGlow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="0" stdDeviation="3" flood-color="#d71920" flood-opacity="0.8"/>
          </filter>
        </defs>
        <!-- Pulsing Bounding Box -->
        <rect x="${x}" y="${y}" width="${w}" height="${h}"
              fill="rgba(215, 25, 32, 0.18)"
              stroke="#d71920"
              stroke-width="3"
              stroke-dasharray="6 3"
              rx="4"
              filter="url(#evidenceGlow)">
          <animate attributeName="stroke-dashoffset" values="0;18" dur="1.2s" repeatCount="indefinite"/>
        </rect>
        <!-- High Contrast Corner Markers -->
        <path d="M ${x} ${y + 12} L ${x} ${y} L ${x + 12} ${y}
                 M ${x + w - 12} ${y} L ${x + w} ${y} L ${x + w} ${y + 12}
                 M ${x} ${y + h - 12} L ${x} ${y + h} L ${x + 12} ${y + h}
                 M ${x + w - 12} ${y + h} L ${x + w} ${y + h} L ${x + w} ${y + h - 12}"
              stroke="#000000" stroke-width="2.5" fill="none"/>
        <!-- Crosshair Centroid -->
        <line x1="${x + w/2 - 7}" y1="${y + h/2}" x2="${x + w/2 + 7}" y2="${y + h/2}" stroke="#d71920" stroke-width="2"/>
        <line x1="${x + w/2}" y1="${y + h/2 - 7}" x2="${x + w/2}" y2="${y + h/2 + 7}" stroke="#d71920" stroke-width="2"/>
        <!-- Label Chip in SVG -->
        <rect x="${x}" y="${Math.max(0, y - 22)}" width="${tagWidth}" height="20" fill="#000000" rx="3"/>
        <text x="${x + 6}" y="${Math.max(14, y - 8)}" fill="#ffffff" font-size="10.5" font-family="-apple-system, BlinkMacSystemFont, sans-serif" font-weight="700">${cType}</text>
      `;
    }

    // 4. Update Right Information Column with Evidence Explanation Callout
    if (panelEvidenceDetails) {
      panelEvidenceDetails.style.display = "block";
      if (btnInspectEvidence) btnInspectEvidence.textContent = "Evidence Details ↑";

      let callout = document.getElementById("active-evidence-callout-box");
      if (!callout) {
        callout = document.createElement("div");
        callout.id = "active-evidence-callout-box";
        callout.className = "active-evidence-callout";
        panelEvidenceDetails.insertBefore(callout, panelEvidenceDetails.firstChild);
      }
      callout.innerHTML = `
        <strong>Selected Candidate: ${cType}</strong>
        <div class="ev-desc">${getEvidenceExplanation(cType)}</div>
        <div class="ev-coords">Spatial Centroid: [X: ${Math.round(x + w/2)}, Y: ${Math.round(y + h/2)}] &bull; Bounding Box: ${w}&times;${h} px &bull; Candidate Evidence</div>
      `;
    }
  }

  function clearEvidenceHighlight() {
    if (evidenceSvgOverlay) {
      evidenceSvgOverlay.innerHTML = "";
      evidenceSvgOverlay.style.display = "none";
    }
    if (activeEvidenceIndicator) {
      activeEvidenceIndicator.style.display = "none";
    }
    document.querySelectorAll(".crop-card").forEach(c => c.classList.remove("active-crop"));
    const callout = document.getElementById("active-evidence-callout-box");
    if (callout) callout.remove();
  }

  if (btnClearEvidence) {
    btnClearEvidence.addEventListener("click", clearEvidenceHighlight);
  }

  function getEvidenceExplanation(type) {
    const t = (type || "").toLowerCase();
    if (t.includes("microaneurysm")) {
      return "Focal capillary dilatations resulting from pericyte loss. They represent the earliest clinically detectable sign of diabetic retinopathy (ICDR Grade 1).";
    } else if (t.includes("hard exudate") || t.includes("exudate")) {
      return "Lipid and lipoprotein precipitates within the retinal parenchyma caused by breakdown of the blood-retinal barrier and persistent capillary leakage.";
    } else if (t.includes("hemorrhage")) {
      return "Intraretinal microvascular ruptures situated in the deep retinal layers (dot-and-blot) or superficial nerve fiber layer (flame-shaped).";
    } else if (t.includes("soft exudate") || t.includes("cotton")) {
      return "Cotton-wool spots representing focal retinal nerve fiber layer ischemia and axoplasmic flow obstruction.";
    }
    return "Candidate microvascular finding detected by feature extraction algorithms. Independent clinical confirmation is required.";
  }

  // =========================================================================
  // 7. On-Demand Region Sensitivity Testing
  // =========================================================================
  if (btnRunSensitivity) {
    btnRunSensitivity.addEventListener("click", async () => {
      if (!currentAnalysis) {
        alert("Please run an analysis first to obtain candidate evidence regions.");
        return;
      }

      btnRunSensitivity.disabled = true;
      btnRunSensitivity.textContent = "Evaluating Sensitivity...";
      if (sensitivityResultsWrap) {
        sensitivityResultsWrap.style.display = "block";
        sensitivityResultsWrap.innerHTML = '<div class="small-text" style="padding: 6px; color: var(--text-muted);">Masking candidate regions and re-running EXP-001 forward pass without weight modification...</div>';
      }

      try {
        const payload = {
          image_id: currentAnalysis.image_id,
          target_class: selectTargetClass ? (selectTargetClass.value === "predicted" ? null : parseInt(selectTargetClass.value, 10)) : null
        };

        const resp = await fetch("/api/region-sensitivity", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });

        if (!resp.ok) {
          const err = await resp.json();
          throw new Error(err.detail || "Failed to evaluate region sensitivity.");
        }

        const data = await resp.json();
        renderRegionSensitivityResults(data);
      } catch (err) {
        if (sensitivityResultsWrap) {
          sensitivityResultsWrap.innerHTML = `<div class="small-text" style="color: #b91c1c;">Region sensitivity analysis failed: ${err.message}</div>`;
        }
      } finally {
        btnRunSensitivity.disabled = false;
        btnRunSensitivity.textContent = "Test Sensitivity";
      }
    });
  }

  function renderRegionSensitivityResults(data) {
    if (!sensitivityResultsWrap) return;

    if (!data.results || data.results.length === 0) {
      sensitivityResultsWrap.innerHTML = `
        <div class="small-text" style="color: var(--text-muted); padding: 6px;">
          ${data.overall_interpretation || "No distinct candidate lesion regions identified for on-demand sensitivity masking."}
        </div>
      `;
      return;
    }

    let html = `
      <table class="sensitivity-table">
        <thead>
          <tr>
            <th>Masked Candidate Region</th>
            <th>Target Class</th>
            <th>Original Prob</th>
            <th>Masked Prob</th>
            <th>Delta</th>
            <th>Analysis Statement</th>
          </tr>
        </thead>
        <tbody>
    `;

    data.results.forEach(r => {
      const deltaClass = r.prob_delta > 0.005 ? "delta-positive" : (r.prob_delta < -0.005 ? "delta-negative" : "delta-neutral");
      const sign = r.prob_delta > 0 ? "+" : "";
      html += `
        <tr>
          <td><strong>${r.region_name}</strong> (${r.region_type})</td>
          <td>Grade ${r.target_class}</td>
          <td>${(r.original_prob * 100).toFixed(1)}%</td>
          <td>${(r.masked_prob * 100).toFixed(1)}%</td>
          <td class="${deltaClass}">${sign}${r.delta_percent}%</td>
          <td class="small-text">${r.statement}</td>
        </tr>
      `;
    });

    html += `
        </tbody>
      </table>
      <div class="small-text" style="margin-top: 6px; font-style: italic; color: var(--text-muted);">
        Notice: ${data.overall_interpretation}
      </div>
    `;

    sensitivityResultsWrap.innerHTML = html;
  }

  // =========================================================================
  // 8. Grounded ExplainAI Q&A Accordion (8 Mandated Sections)
  // =========================================================================
  function renderExplainAI(explanation) {
    if (!explainaiQaList) return;

    if (!explanation || !explanation.questions || explanation.questions.length === 0) {
      explainaiQaList.innerHTML = '<div class="empty-state-text">Run screening analysis to generate evidence-grounded clinical explanations.</div>';
      return;
    }

    explainaiQaList.innerHTML = "";

    const sectionTitles = [
      { title: "01 Image Quality", sub: "Optical Validation & Quality Gate" },
      { title: "02 Processing", sub: "CLAHE Enhancement & Preprocessing" },
      { title: "03 Model Prediction", sub: "EXP-001 Inference & ICDR Grade" },
      { title: "04 Visual Evidence", sub: "Candidate Lesion Findings" },
      { title: "05 Model Attribution", sub: "Grad-CAM++ Spatial Saliency" },
      { title: "06 Region Sensitivity", sub: "Perturbation & Candidate Masking" },
      { title: "07 Uncertainty", sub: "Decision Margins & 5-Class Probabilities" },
      { title: "08 Limitations", sub: "Clinical Scope & Prototype Constraints" }
    ];

    explanation.questions.forEach((q, idx) => {
      const meta = sectionTitles[idx] || { title: `0${idx + 1} Section`, sub: "Screening Detail" };
      const item = document.createElement("div");
      item.className = "qa-item";
      // Auto-expand first 2 items by default
      if (idx < 2) item.classList.add("expanded");

      const grounding = (q.grounding_fields && q.grounding_fields.length > 0)
        ? `<div class="qa-meta">Grounded in: ${q.grounding_fields.join(", ")}</div>`
        : "";

      item.innerHTML = `
        <div class="qa-q" role="button" tabindex="0">
          <div>
            <div style="font-size: 10px; font-weight: 800; text-transform: uppercase; color: var(--accent-red);">${meta.title}</div>
            <div style="font-weight: 700; color: #000000; margin-top: 1px;">${q.question}</div>
          </div>
          <span class="qa-icon">&#9656;</span>
        </div>
        <div class="qa-a">
          <p>${q.answer}</p>
          ${grounding}
        </div>
      `;

      const header = item.querySelector(".qa-q");
      header.addEventListener("click", () => item.classList.toggle("expanded"));

      explainaiQaList.appendChild(item);
    });
  }

  // =========================================================================
  // 9. Layer Toggles & Crossfade Layer Switching
  // =========================================================================
  document.querySelectorAll(".layer-tab-group .toggle-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const layer = btn.getAttribute("data-layer");
      setActiveLayer(layer);
    });
  });

  function setActiveLayer(layer) {
    document.querySelectorAll(".layer-tab-group .toggle-btn").forEach(b => {
      b.classList.toggle("active", b.getAttribute("data-layer") === layer);
    });

    const overlayUrl = currentOverlays[layer] || (layer === "evidence" ? (currentOverlays["lesions"] || currentOverlays["vessels"]) : null);
    if (overlayUrl && mainImageView) {
      mainImageView.style.opacity = "0.2";
      setTimeout(() => {
        mainImageView.src = overlayUrl;
        mainImageView.style.opacity = "1";
      }, 80);
    }

    // Show attribution target selector only on Attribution (heatmap) tab
    if (attributionTargetSelector) {
      attributionTargetSelector.style.display = (layer === "heatmap") ? "flex" : "none";
    }

    // Color legend
    if (layer === "evidence" || layer === "lesions") {
      layerLegend.style.display = "flex";
      layerLegend.innerHTML = `
        <span class="legend-item"><span class="legend-swatch" style="background:#ffe600;"></span> Hard Exudates</span>
        <span class="legend-item"><span class="legend-swatch" style="background:#ff00ea;"></span> Microaneurysms</span>
        <span class="legend-item"><span class="legend-swatch" style="background:#ff2a2a;"></span> Hemorrhages</span>
        <span style="color:var(--text-muted); font-size:10px;">(Candidate lesion findings)</span>
      `;
    } else if (layer === "heatmap") {
      layerLegend.style.display = "flex";
      layerLegend.innerHTML = `
        <span class="legend-item"><span class="legend-swatch" style="background:linear-gradient(to right, #1e1b4b, #047857, #facc15, #dc2626);"></span> Model Attribution (Grad-CAM++)</span>
        <span style="color:var(--text-muted); font-size:10px;">Neural decision regions (Not lesion segmentation)</span>
      `;
    } else if (layer === "enhanced") {
      layerLegend.style.display = "flex";
      layerLegend.innerHTML = `
        <span class="legend-item"><span class="legend-swatch" style="background:#000000;"></span> Green-channel CLAHE (Optimized hemoglobin contrast)</span>
      `;
    } else {
      layerLegend.style.display = "none";
    }
  }

  // =========================================================================
  // 10. Focus Evidence Mode Toggle
  // =========================================================================
  function toggleFocusMode() {
    if (!workstationShell) return;
    isFocusMode = !isFocusMode;
    workstationShell.classList.toggle("focus-mode-active", isFocusMode);
    if (btnFocusMode) {
      btnFocusMode.innerHTML = isFocusMode
        ? '<span class="focus-icon">&times;</span> Exit Focus (ESC)'
        : '<span class="focus-icon">&#9974;</span> Focus Evidence';
    }
  }

  if (btnFocusMode) {
    btnFocusMode.addEventListener("click", toggleFocusMode);
  }

  // =========================================================================
  // 11. Inspect Detail Button Toggles
  // =========================================================================
  if (btnInspectQuality && panelQualityDetails) {
    btnInspectQuality.addEventListener("click", () => {
      const isHidden = panelQualityDetails.style.display === "none";
      panelQualityDetails.style.display = isHidden ? "block" : "none";
      btnInspectQuality.textContent = isHidden ? "Quality Details ↑" : "Quality Details →";
    });
  }

  if (btnInspectEvidence && panelEvidenceDetails) {
    btnInspectEvidence.addEventListener("click", () => {
      const isHidden = panelEvidenceDetails.style.display === "none";
      panelEvidenceDetails.style.display = isHidden ? "block" : "none";
      btnInspectEvidence.textContent = isHidden ? "Evidence Details ↑" : "Evidence Details →";
    });
  }

  if (btnInspectTrace && panelTraceDetails) {
    btnInspectTrace.addEventListener("click", () => {
      const isHidden = panelTraceDetails.style.display === "none";
      panelTraceDetails.style.display = isHidden ? "block" : "none";
      btnInspectTrace.textContent = isHidden ? "Trace Details ↑" : "Trace Details →";
    });
  }

  // =========================================================================
  // 12. ExplainAI "Why this result?" Slide-Over Drawer
  // =========================================================================
  if (btnOpenExplain && explainaiDrawerOverlay) {
    btnOpenExplain.addEventListener("click", () => {
      explainaiDrawerOverlay.style.display = "flex";
    });
  }

  if (btnCloseExplain && explainaiDrawerOverlay) {
    btnCloseExplain.addEventListener("click", () => {
      explainaiDrawerOverlay.style.display = "none";
    });
  }

  if (explainaiDrawerOverlay) {
    explainaiDrawerOverlay.addEventListener("click", (e) => {
      if (e.target === explainaiDrawerOverlay) {
        explainaiDrawerOverlay.style.display = "none";
      }
    });
  }

  // =========================================================================
  // 13. Pipeline Transparency "Why?" Modals
  // =========================================================================
  async function showStageModal(stageId) {
    try {
      const resp = await fetch(`/api/pipeline-stages/${stageId}`);
      if (!resp.ok) {
        throw new Error("Pipeline stage details not available.");
      }
      const data = await resp.json();

      modalStageCategory.textContent = data.category || "Pipeline Stage";
      modalStageName.textContent = data.name || stageId;
      modalStageWhat.textContent = data.what_it_does || "--";
      modalStageWhy.textContent = data.why_used || "--";
      modalStageParams.textContent = JSON.stringify(data.actual_parameters || {}, null, 2);
      modalStageChanged.textContent = data.what_changed_in_image || "--";
      modalStageLimitation.textContent = data.limitation || "--";

      stageModal.style.display = "flex";
    } catch (err) {
      alert("Error loading stage info: " + err.message);
    }
  }

  // Wire all "Why?" and stage triggers
  document.querySelectorAll(".btn-why, .btn-why-text, .flow-stage").forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const stageId = btn.getAttribute("data-stage");
      if (stageId) showStageModal(stageId);
    });
  });

  if (btnStages) {
    btnStages.addEventListener("click", () => {
      showStageModal("stage_1_quality_gate");
    });
  }

  if (btnCloseStage) {
    btnCloseStage.addEventListener("click", () => {
      stageModal.style.display = "none";
    });
  }

  // =========================================================================
  // 14. Research Validation & About Modals & Keyboard Shortcuts
  // =========================================================================
  if (btnAbout && aboutModal && btnCloseAbout) {
    btnAbout.addEventListener("click", () => aboutModal.style.display = "flex");
    btnCloseAbout.addEventListener("click", () => aboutModal.style.display = "none");
  }

  if (btnValidation && validationModal && btnCloseValidation) {
    btnValidation.addEventListener("click", () => validationModal.style.display = "flex");
    btnCloseValidation.addEventListener("click", () => validationModal.style.display = "none");
  }

  window.addEventListener("click", (e) => {
    if (e.target === aboutModal) aboutModal.style.display = "none";
    if (e.target === validationModal) validationModal.style.display = "none";
    if (e.target === stageModal) stageModal.style.display = "none";
  });

  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (isFocusMode) {
        toggleFocusMode();
      }
      if (aboutModal) aboutModal.style.display = "none";
      if (validationModal) validationModal.style.display = "none";
      if (stageModal) stageModal.style.display = "none";
      if (explainaiDrawerOverlay) explainaiDrawerOverlay.style.display = "none";
    }
  });

  // =========================================================================
  // 15. Report Linking
  // =========================================================================
  function setupReportLinks(report) {
    reportActionBar.style.display = "flex";
    btnViewReport.href = report.relative_url;
    const pdfFilename = report.pdf_filename || report.filename.replace(".html", ".pdf");
    btnDownloadReport.href = `/api/download-report/${pdfFilename}`;
    btnDownloadReport.setAttribute("download", pdfFilename);
  }

  // Initial metadata sync
  updateMetadataChips("IDLE");
});
