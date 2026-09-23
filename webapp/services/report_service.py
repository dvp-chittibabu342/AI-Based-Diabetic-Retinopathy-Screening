"""
report_service.py
=================
Generates professional, printable, self-contained HTML clinical screening reports.

RetinaGuard 2.0 Compliance:
1. Sourced exclusively from the unified AnalysisResult object.
2. Displays candidate evidence framing (candidates, crops, bounding boxes).
3. Displays complete 5-class model probability distribution and margin.
4. Identifies referral rule as "Configured prototype screening rule (Grade >= 2)".
5. Includes grounded ExplainAI Q&A summary.
6. Prominent disclaimers: AI-assisted screening decision support, not an autonomous medical diagnosis.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import os
import re
import shutil
import base64
import subprocess


class ReportService:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        result: Dict[str, Any],
        patient_id: str = "PT-82910",
        exam_id: str = "EX-00412",
        eye: str = "Right (OD)"
    ) -> Dict[str, str]:
        date_str = datetime.now().strftime("%d-%b-%Y %H:%M")
        timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        meta = result.get("metadata") or {}
        inp = result.get("input") or {}
        q = result.get("quality") or {}
        enh = result.get("preprocessing") or result.get("enhancement") or {}
        st = result.get("anatomy") or result.get("structures") or {}
        l = result.get("lesion_candidates") or result.get("lesions") or {}
        g = result.get("model") or result.get("grading") or {}
        x = result.get("xai") or {}
        exp = result.get("explanation") or {}

        patient_id = inp.get("patient_id") or patient_id
        exam_id = inp.get("exam_id") or exam_id
        eye = inp.get("eye") or eye
        analysis_id = meta.get("analysis_id", f"RG2-{timestamp_slug}")
        is_ungradable = not q.get("gradable", True)

        # Grade color mapping
        grade = g.get("predicted_grade") if "predicted_grade" in g else g.get("grade", -1)
        colors = {
            0: "#1a7f37",
            1: "#b08800",
            2: "#b08800",
            3: "#cf222e",
            4: "#82071e",
            -1: "#6e7781"
        }
        grade_color = colors.get(grade, "#6e7781")
        sev_name = g.get("severity_name", "N/A")

        # HTML Head and Styles
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RetinaGuard 2.0 Screening Report — {exam_id}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f8fafc; color: #0f172a; font-size: 13px; line-height: 1.5; }}
  .sheet {{ max-width: 880px; margin: 24px auto; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; overflow: hidden; box-shadow: 0 4px 16px rgba(0,0,0,0.06); }}
  .header {{ background: #000000; color: #ffffff; padding: 22px 32px; display: flex; justify-content: space-between; align-items: center; border-bottom: 3px solid #d97706; }}
  .header h1 {{ font-size: 22px; font-weight: 700; letter-spacing: 0.5px; }}
  .header .sub {{ font-size: 11px; color: #f59e0b; text-transform: uppercase; letter-spacing: 1px; margin-top: 3px; font-weight: 600; }}
  .header .meta {{ text-align: right; font-size: 11px; color: #94a3b8; line-height: 1.4; }}
  .header .meta strong {{ color: #f59e0b; }}
  .section {{ padding: 18px 32px; border-bottom: 1px solid #f1f5f9; }}
  .sec-title {{ font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: #d97706; margin-bottom: 12px; }}
  .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .kv-label {{ font-size: 10px; text-transform: uppercase; color: #64748b; font-weight: 600; }}
  .kv-val {{ font-size: 13px; font-weight: 600; color: #000000; margin-top: 2px; }}
  .badge-good {{ background: #ecfdf5; color: #059669; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; border: 1px solid #a7f3d0; }}
  .badge-ungradable {{ background: #fef2f2; color: #dc2626; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; border: 1px solid #fecaca; }}
  .badge-borderline {{ background: #fffbeb; color: #d97706; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; border: 1px solid #fde68a; }}
  .grade-card {{ background: #ffffff; border: 1px solid #e2e8f0; border-left: 4px solid #d97706; border-radius: 6px; padding: 16px; text-align: center; margin-bottom: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.03); }}
  .grade-big {{ font-size: 30px; font-weight: 800; color: {grade_color}; }}
  .grade-name {{ font-size: 15px; font-weight: 700; margin-top: 4px; color: #000000; }}
  .conf-bar {{ display: flex; align-items: center; justify-content: center; gap: 8px; font-size: 12px; color: #475569; margin-top: 6px; }}
  .referral-box {{ background: #fffbeb; border-left: 4px solid #d97706; padding: 12px 16px; border-radius: 0 4px 4px 0; margin-top: 10px; }}
  .referral-title {{ font-size: 11px; font-weight: 700; text-transform: uppercase; color: #b45309; margin-bottom: 4px; letter-spacing: 0.5px; }}
  .table {{ width: 100%; border-collapse: collapse; margin-top: 6px; font-size: 12px; }}
  .table th {{ text-align: left; padding: 7px 10px; background: #f8fafc; border-bottom: 2px solid #e2e8f0; font-size: 11px; color: #475569; font-weight: 600; }}
  .table td {{ padding: 7px 10px; border-bottom: 1px solid #f1f5f9; color: #1e293b; }}
  .img-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin-top: 10px; }}
  .img-card {{ border: 1px solid #e2e8f0; border-radius: 4px; overflow: hidden; background: #000000; text-align: center; }}
  .img-card img {{ width: 100%; height: 160px; object-fit: contain; display: block; }}
  .img-card .caption {{ background: #f8fafc; padding: 6px; font-size: 10px; font-weight: 600; color: #475569; border-top: 1px solid #e2e8f0; }}
  .crop-grid {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; }}
  .crop-card {{ border: 1px solid #e2e8f0; border-radius: 4px; padding: 6px; background: #ffffff; text-align: center; width: 110px; }}
  .crop-card img {{ width: 96px; height: 96px; object-fit: cover; display: block; border-radius: 2px; margin-bottom: 4px; }}
  .crop-card .c-type {{ font-size: 9px; font-weight: 700; color: #000000; line-height: 1.2; }}
  .qa-block {{ background: #ffffff; border: 1px solid #e2e8f0; border-left: 3px solid #d97706; border-radius: 4px; padding: 12px 16px; margin-bottom: 8px; }}
  .qa-q {{ font-weight: 700; font-size: 12px; color: #000000; margin-bottom: 4px; }}
  .qa-a {{ font-size: 12px; color: #334155; line-height: 1.5; }}
  .disclaimer {{ background: #fffbeb; border-left: 4px solid #d97706; padding: 12px 16px; font-size: 11px; color: #78350f; margin: 18px 32px; border-radius: 0 4px 4px 0; }}
  .footer {{ background: #000000; color: #94a3b8; padding: 14px 32px; text-align: center; font-size: 11px; border-top: 1px solid #1f2937; }}
  @media print {{ body {{ background: #fff; }} .sheet {{ border: none; box-shadow: none; margin: 0; width: 100%; }} }}
</style>
</head>
<body>

<div class="sheet">
  <div class="header">
    <div>
      <h1>RetinaGuard 2.0</h1>
      <div class="sub">AI-Assisted Retinal Screening Workstation &bull; Evidence &amp; Explainability</div>
    </div>
    <div class="meta">
      <div><strong>Report ID:</strong> {analysis_id}</div>
      <div><strong>Generated:</strong> {date_str}</div>
      <div><strong>Model:</strong> EXP-001 (EfficientNet-B0)</div>
    </div>
  </div>

  <div class="section">
    <div class="sec-title">Examination &amp; Patient Information</div>
    <div class="grid-3">
      <div><div class="kv-label">Patient Identifier</div><div class="kv-val">{patient_id}</div></div>
      <div><div class="kv-label">Examination ID</div><div class="kv-val">{exam_id}</div></div>
      <div><div class="kv-label">Eye Orientation</div><div class="kv-val">{eye}</div></div>
    </div>
  </div>

  <div class="section">
    <div class="sec-title">1. Image Quality Assessment</div>
    <div class="grid-2">
      <div>
        <div class="kv-label">Quality Status</div>
        <div class="kv-val">
          <span class="badge-{q.get('status', 'ungradable').lower()}">{q.get('status', 'UNGRADABLE')}</span>
          &nbsp;&bull;&nbsp; Quality Score: <strong>{q.get('score', 0)} / 100</strong>
        </div>
        <div style="font-size: 11px; color: #57606a; margin-top: 6px;">
          <strong>Finding:</strong> {q.get('reason', 'N/A')}
        </div>
        <div style="font-size: 10px; color: #6e7781; margin-top: 4px;">
          {q.get('threshold_notes', 'Configured prototype threshold: minimum score 40.0.')}
        </div>
      </div>
      <div>
        <table class="table" style="margin: 0;">
          <tr><td>Focus / Sharpness</td><td><strong>{q.get('focus', 'N/A')}</strong> (Laplacian Var: {q.get('laplacian_var', 'N/A')})</td></tr>
          <tr><td>Illumination</td><td><strong>{q.get('illumination', 'N/A')}</strong> (Mean: {q.get('mean_lum', 'N/A')})</td></tr>
          <tr><td>Contrast &amp; FOV</td><td><strong>{q.get('contrast', 'N/A')}</strong> &bull; FOV: {q.get('fov_percent', 'N/A')}%</td></tr>
        </table>
      </div>
    </div>
  </div>
"""

        # Section 2: If Ungradable -> Safety Gate Banner
        if is_ungradable:
            html += f"""
  <div class="section" style="background: #fff5f5;">
    <div class="sec-title" style="color: #cf222e;">Safety Gate Triggered &mdash; Downstream Grading Halted</div>
    <div style="padding: 14px; border: 1px solid #ffdcd7; border-radius: 6px; background: #ffebe9; color: #82071e;">
      <strong>DOWNSTREAM GRADING HALTED:</strong> Retinal image quality is insufficient for diagnostic screening.
      To prevent medical misdiagnosis, automated DR classification was safely prevented.
      <div style="margin-top: 8px; font-weight: 600;">Action Required: {q.get('recommendation', 'Recapture image with improved focus.')}</div>
    </div>
  </div>
"""
        else:
            # Section 2: DR Severity & Probabilities
            probs = g.get("probabilities", [])
            prob_rows = ""
            for p in probs:
                p_cls = "background: rgba(217, 119, 6, 0.08); font-weight: 700; color: #000;" if p.get("grade") == grade else ""
                prob_rows += f"""<tr style="{p_cls}">
                  <td>Grade {p.get('grade')} &bull; {p.get('label')}</td>
                  <td>{p.get('probability', 0.0):.4f}</td>
                  <td>{p.get('percentage', 0.0)}%</td>
                </tr>"""

            html += f"""
  <div class="section">
    <div class="sec-title">2. Diabetic Retinopathy Screening Result</div>
    <div class="grade-card">
      <div style="font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 700; letter-spacing: 0.5px;">AI Screening Severity</div>
      <div class="grade-big">Grade {grade}</div>
      <div class="grade-name">{sev_name}</div>
      <div class="conf-bar">
        Model Probability: <strong>{g.get('confidence_percent', 0)}%</strong> &bull;
        Decision Margin: <strong>{g.get('margin_percent', 0)}%</strong> over {g.get('runner_up_name', 'runner-up')} &bull;
        Status: <strong>{g.get('referral_badge', 'N/A')}</strong>
      </div>
      <div style="font-size: 10px; color: #64748b; margin-top: 4px;">
        {g.get('calibration_status', 'Calibration: not applied (raw model probabilities)')}
      </div>
    </div>

    <div class="referral-box">
      <div class="referral-title">Clinical Screening Triage Recommendation</div>
      <div style="font-weight: 600; color: #92400e;">{g.get('recommendation', 'N/A')}</div>
      <div style="font-size: 10px; color: #64748b; margin-top: 4px;">
        Rule applied: {g.get('referral_rule', 'Configured prototype screening rule (Grade >= 2)')}
      </div>
    </div>

    <div style="margin-top: 14px;">
      <div class="kv-label" style="margin-bottom: 4px;">Complete 5-Class Output Probability Distribution</div>
      <table class="table">
        <thead><tr><th>ICDR Severity Level</th><th>Probability</th><th>Percentage</th></tr></thead>
        <tbody>{prob_rows}</tbody>
      </table>
    </div>
  </div>

  <div class="section">
    <div class="sec-title">3. Quantitative Candidate Evidence &amp; Landmarks</div>
    <div class="grid-2">
      <div>
        <div style="font-weight: 700; font-size: 11px; margin-bottom: 4px; color: #57606a;">Retinal Anatomical Landmarks</div>
        <table class="table">
          <tr><td>Optic Disc</td><td>Centroid: {st.get('optic_disc', {}).get('centroid', 'N/A')}, Radius: {st.get('optic_disc', {}).get('radius_px', 'N/A')} px</td></tr>
          <tr><td>Fovea Centralis</td><td>Coordinates: {st.get('fovea', {}).get('coordinates', 'N/A')}</td></tr>
          <tr><td>Retinal Vasculature</td><td>Vascular Density: <strong>{st.get('vessels', {}).get('density_percent', 'N/A')}%</strong></td></tr>
        </table>
      </div>
      <div>
        <div style="font-weight: 700; font-size: 11px; margin-bottom: 4px; color: #57606a;">Candidate Lesion Evidence</div>
        <table class="table">
          <tr><td>Microaneurysms</td><td><strong>{l.get('microaneurysms', {}).get('candidate_count', 0)}</strong> candidates ({l.get('microaneurysms', {}).get('total_area_px', 0)} px)</td></tr>
          <tr><td>Hard Exudates</td><td><strong>{l.get('hard_exudates', {}).get('candidate_count', 0)}</strong> candidates (Disc-excluded mask)</td></tr>
          <tr><td>Hemorrhages</td><td><strong>{l.get('hemorrhages', {}).get('candidate_count', 0)}</strong> candidates ({l.get('hemorrhages', {}).get('total_area_px', 0)} px)</td></tr>
          <tr><td>NV Risk Indicator</td><td><strong>{l.get('neovascularization', {}).get('risk_level', 'Low')}</strong> ({l.get('neovascularization', {}).get('description', '')})</td></tr>
        </table>
      </div>
    </div>
    <div style="font-size: 10px; color: #656d76; margin-top: 6px; font-style: italic;">
      Notice: Findings represent candidate evidence identified by morphological analysis and U-Net segmentation. They are not independently confirmed clinical lesions.
    </div>
"""

            # Candidate crops if present
            crops = l.get("candidate_crops", [])
            if crops:
                crop_cards = ""
                for c in crops:
                    crop_cards += f"""<div class="crop-card">
                      <img src="{c.get('crop_url', '')}" alt="{c.get('type', '')}">
                      <div class="c-type">{c.get('type', 'Candidate')}</div>
                      <div style="font-size: 8px; color: #656d76;">{c.get('area_px', 0)} px</div>
                    </div>"""

                html += f"""
    <div style="margin-top: 12px;">
      <div class="kv-label">Candidate Evidence Crops</div>
      <div class="crop-grid">{crop_cards}</div>
    </div>
"""

            # Multi-layer images
            orig_url = inp.get("original_url") or result.get("original_url", "")
            enh_url = enh.get("enhanced_url") or enh.get("relative_url", "")
            vess_url = st.get("vessel_overlay_url", "")
            lesion_url = l.get("overlay_url", "")
            xai_url = x.get("gradcam_url") or g.get("gradcam_url", "")
            xai_name = x.get("method", "Grad-CAM++")

            html += f"""
  </div>

  <div class="section">
    <div class="sec-title">4. Multi-Layer Visual Evidence &amp; Model Attribution</div>
    <div class="img-grid">
      <div class="img-card">
        <img src="{orig_url}" alt="Original Fundus Scan">
        <div class="caption">Original Fundus Photograph</div>
      </div>
      <div class="img-card">
        <img src="{enh_url}" alt="Enhanced CLAHE View">
        <div class="caption">CLAHE Enhanced (Green Channel)</div>
      </div>
      <div class="img-card">
        <img src="{vess_url}" alt="Vessels &amp; Landmarks">
        <div class="caption">Landmarks &amp; Vasculature</div>
      </div>
      <div class="img-card">
        <img src="{lesion_url}" alt="Candidate Lesion Map">
        <div class="caption">Candidate Lesion Evidence</div>
      </div>
      <div class="img-card">
        <img src="{xai_url}" alt="Model Attribution Map">
        <div class="caption">{xai_name} (Model Attribution)</div>
      </div>
    </div>
    <div style="font-size: 10px; color: #656d76; margin-top: 6px; font-style: italic;">
      Notice: Model attribution visualization highlights spatial regions that contributed to the neural network score; it is not manual lesion segmentation.
    </div>
  </div>
"""

        # Section 5: Grounded ExplainAI Q&A
        questions = exp.get("questions", [])
        if questions:
            qa_html = ""
            for item in questions:
                qa_html += f"""<div class="qa-block">
                  <div class="qa-q">{item.get('question')}</div>
                  <div class="qa-a">{item.get('answer')}</div>
                </div>"""

            html += f"""
  <div class="section">
    <div class="sec-title">5. ExplainAI &mdash; Grounded Screening Reasoning</div>
    {qa_html}
  </div>
"""

        # Section 6: Disclaimers & Footer
        html += f"""
  <div class="disclaimer">
    <strong>IMPORTANT CLINICAL DISCLAIMER:</strong> This report was generated by RetinaGuard 2.0, an AI-assisted diabetic retinopathy screening decision-support research prototype. It does NOT provide an autonomous medical diagnosis. All findings, candidate lesion counts, probability values, and recommendations must be confirmed by a licensed ophthalmologist or retina specialist before clinical management decisions are undertaken.
  </div>

  <div class="footer">
    RetinaGuard 2.0 &mdash; AI-Assisted Diabetic Retinopathy Screening Workstation &bull; SIH 2026 &bull; Team VyuhaX
  </div>
</div>

</body>
</html>"""

        filename = f"report_{exam_id}_{timestamp_slug}.html"
        out_path = self.output_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)

        pdf_filename = f"report_{exam_id}_{timestamp_slug}.pdf"
        pdf_path = self.output_dir / pdf_filename
        self.convert_html_to_pdf(out_path, pdf_path)

        return {
            "filename": filename,
            "file_path": str(out_path),
            "filepath": str(out_path),
            "relative_url": f"/outputs/{filename}",
            "pdf_filename": pdf_filename,
            "pdf_path": str(pdf_path) if pdf_path.exists() else None,
            "pdf_url": f"/outputs/{pdf_filename}" if pdf_path.exists() else None
        }

    @staticmethod
    def find_browser_executable() -> Optional[str]:
        candidates = [
            os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
            os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
            os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%LocalAppData%\Microsoft\Edge\Application\msedge.exe"),
            shutil.which("msedge"),
            shutil.which("chrome"),
            shutil.which("chromium"),
        ]
        for c in candidates:
            if c and os.path.exists(c):
                return c
        return None

    def convert_html_to_pdf(self, html_path: Path, pdf_path: Path) -> bool:
        """
        Converts an existing HTML clinical report to a high-fidelity PDF document.
        Embeds local image assets as base64 data URIs so that all images render reliably.
        """
        browser_exe = self.find_browser_executable()
        if not browser_exe:
            print("[ReportService] Error: No compatible headless browser (Edge/Chrome) found for PDF generation.")
            return False

        if not html_path.exists():
            print(f"[ReportService] Error: Source HTML report not found at {html_path}")
            return False

        try:
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            # Embed image assets as base64 so headless renderer doesn't fail on relative URLs
            def _embed_img(match):
                src = match.group(1)
                local_file = None
                if src.startswith("/outputs/"):
                    local_file = self.output_dir / src[len("/outputs/"):]
                elif src.startswith("/demo/sample_images/"):
                    sub = src[len("/demo/sample_images/"):]
                    for parent in [
                        Path("sample_images"),
                        Path("webapp/static/sample_images"),
                        Path("webapp/sample_images")
                    ]:
                        cand = parent / sub
                        if cand.exists():
                            local_file = cand
                            break
                elif Path(src).exists():
                    local_file = Path(src)

                if local_file and local_file.exists():
                    try:
                        ext = local_file.suffix.lower().replace(".", "")
                        mime = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"
                        with open(local_file, "rb") as img_f:
                            b64 = base64.b64encode(img_f.read()).decode("utf-8")
                        return f'src="data:{mime};base64,{b64}"'
                    except Exception:
                        pass
                return f'src="{src}"'

            standalone_html = re.sub(r'src=["\']([^"\']+)["\']', _embed_img, html_content)
            temp_html_path = self.output_dir / f"temp_{pdf_path.stem}.html"
            with open(temp_html_path, "w", encoding="utf-8") as f:
                f.write(standalone_html)

            cmd = [
                browser_exe,
                "--headless=new",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--print-to-pdf={str(pdf_path.resolve())}",
                str(temp_html_path.resolve())
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            # Cleanup temp HTML
            if temp_html_path.exists():
                try:
                    temp_html_path.unlink()
                except Exception:
                    pass

            if res.returncode == 0 and pdf_path.exists() and pdf_path.stat().st_size > 0:
                with open(pdf_path, "rb") as f:
                    header = f.read(5)
                if header.startswith(b"%PDF-"):
                    return True

            print(f"[ReportService] PDF conversion failed. Returncode: {res.returncode}, Stderr: {res.stderr}")
            if pdf_path.exists() and pdf_path.stat().st_size == 0:
                pdf_path.unlink()
            return False

        except Exception as e:
            print(f"[ReportService] PDF conversion exception: {e}")
            if pdf_path.exists() and pdf_path.stat().st_size == 0:
                pdf_path.unlink()
            return False

    # Alias for API compatibility
    generate_html_report = generate

