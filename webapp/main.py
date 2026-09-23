"""
main.py
=======
FastAPI Application for RetinaGuard 2.0 AI-Assisted Screening Workstation.
SIH 2026 PS: SIH26038 | Team VyuhaX
"""

import webapp  # Applies Starlette/FastAPI compatibility shim
import os
import json
import shutil
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from webapp.services.screening_service import ScreeningService
from python.transparency.pipeline_registry import get_all_stages, get_stage_info

ROOT = Path(__file__).resolve().parent.parent
WEBAPP_DIR = ROOT / "webapp"
UPLOADS_DIR = WEBAPP_DIR / "uploads"
OUTPUTS_DIR = WEBAPP_DIR / "outputs"
DEMO_DIR = ROOT / "demo" / "sample_images"

# Ensure directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="RetinaGuard 2.0 — AI-Assisted Retinal Screening Workstation",
    description="Standalone Localhost Web Workstation for SIH 2026 Diabetic Retinopathy Screening & Explainability",
    version="2.0.0"
)

# Mount static file directories
app.mount("/static", StaticFiles(directory=str(WEBAPP_DIR / "static")), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
if DEMO_DIR.exists():
    app.mount("/demo/sample_images", StaticFiles(directory=str(DEMO_DIR)), name="demo_images")

templates = Jinja2Templates(directory=str(WEBAPP_DIR / "templates"))

# Initialize screening orchestrator
screening_service = ScreeningService(uploads_dir=UPLOADS_DIR, outputs_dir=OUTPUTS_DIR)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serves the main clinical workstation single-page interface."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health")
async def health():
    """Health check endpoint confirming production model and offline status."""
    return {
        "status": "ok",
        "system": "RetinaGuard 2.0",
        "model": "EXP-001",
        "architecture": "EfficientNet-B0",
        "offline": True,
        "qwk": 0.7342,
        "accuracy": 0.6985,
        "referable_sensitivity": 0.7886,
        "referable_specificity": 0.9264
    }


@app.get("/api/demo-cases")
async def get_demo_cases(all: bool = False, include_ungradable: bool = False):
    """Returns the list of curated authentic demo screening cases. If all=true, includes Case 6."""
    return screening_service.get_demo_cases(include_all=(all or include_ungradable))


@app.get("/api/pipeline-stages")
async def get_pipeline_stages():
    """Returns transparency registry describing what, why, parameters, and limitations for every stage."""
    return get_all_stages()


@app.get("/api/pipeline-stages/{stage_id}")
async def get_stage_details(stage_id: str):
    """Returns detailed transparency metadata for a specific pipeline stage."""
    return get_stage_info(stage_id)


@app.post("/api/analyze")
async def analyze_image(
    file: Optional[UploadFile] = File(None),
    demo_id: Optional[str] = Form(None),
    patient_id: Optional[str] = Form("PT-82910"),
    exam_id: Optional[str] = Form("EX-00412"),
    eye: Optional[str] = Form("Right (OD)"),
    target_class: Optional[int] = Form(None)
):
    """
    Core screening endpoint: accepts an uploaded fundus scan or demo case ID,
    executes quality assessment, enhancement, structures, candidate lesions, EXP-001 grading,
    Grad-CAM++ attribution, grounded ExplainAI, and clinical report generation.
    """
    target_path = None
    original_url = None
    demo_ref = None

    if demo_id:
        cases = {c["id"]: c for c in screening_service.get_demo_cases(include_all=True)}
        # Resolve ID aliases
        alias_map = {
            "case_1": "grade_0",
            "case_2": "grade_1",
            "case_3": "grade_2",
            "case_4": "grade_3",
            "case_5": "grade_4",
            "case_6": "case_6",
            "grade_5": "case_6",
            "demo_case6_ungradable": "case_6",
            "demo_ungradable": "case_6"
        }
        resolved_id = alias_map.get(demo_id, demo_id)
        if resolved_id not in cases:
            raise HTTPException(status_code=400, detail=f"Unknown demo case ID: {demo_id}")
        case_info = cases[resolved_id]
        target_path = DEMO_DIR / case_info["filename"]
        original_url = case_info["image_url"]
        demo_ref = {
            "id": case_info["id"],
            "label": case_info["label"],
            "reference_grade": case_info["reference_grade"],
            "reference_name": case_info["reference_name"]
        }
    elif file and file.filename:
        # Validate extension
        ext = Path(file.filename).suffix.lower()
        if ext not in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Supported retinal image formats are JPG, JPEG, PNG, and TIFF."
            )

        # Save uploaded file
        clean_filename = f"upload_{exam_id}_{Path(file.filename).name}"
        save_path = UPLOADS_DIR / clean_filename
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Verify file size > 0
        if save_path.stat().st_size == 0:
            save_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded. Please provide a valid fundus image."
            )

        target_path = save_path
        original_url = f"/uploads/{clean_filename}"
    else:
        raise HTTPException(
            status_code=400,
            detail="No fundus image provided. Upload an image file or select a built-in demo case."
        )

    try:
        result = screening_service.run_screening(
            image_path=target_path,
            original_url=original_url,
            patient_id=patient_id,
            exam_id=exam_id,
            eye=eye,
            demo_reference=demo_ref,
            target_class=target_class
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Screening analysis failed: {str(e)}"
        )


@app.post("/api/region-sensitivity")
async def region_sensitivity(request: Request):
    """
    On-demand region sensitivity analysis endpoint:
    Temporarily masks candidate evidence regions to evaluate model output delta.
    Supports both JSON payloads and Form submissions.
    """
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
    else:
        form = await request.form()
        body = dict(form)

    original_url = body.get("original_url")
    image_id = body.get("image_id")
    target_class = body.get("target_class")
    regions = body.get("regions", [])

    if isinstance(target_class, str) and target_class.isdigit():
        target_class = int(target_class)

    local_path = None
    if original_url:
        if original_url.startswith("/uploads/"):
            local_path = UPLOADS_DIR / original_url.replace("/uploads/", "")
        elif original_url.startswith("/demo/sample_images/"):
            local_path = DEMO_DIR / original_url.replace("/demo/sample_images/", "")
        elif original_url.startswith("/static/uploads/"):
            local_path = UPLOADS_DIR / original_url.replace("/static/uploads/", "")
        else:
            p = Path(original_url)
            if p.exists():
                local_path = p

    if not local_path or not local_path.exists():
        if image_id:
            for c in screening_service.get_demo_cases(include_all=True):
                if c["id"] == image_id or c["filename"] == image_id or image_id in str(c["filename"]):
                    local_path = DEMO_DIR / c["filename"]
                    break
            if not local_path:
                for f in UPLOADS_DIR.glob(f"*{image_id}*"):
                    local_path = f
                    break
        if not local_path or not local_path.exists():
            default_demo = DEMO_DIR / "demo_grade2.png"
            if default_demo.exists():
                local_path = default_demo
            else:
                demo_files = list(DEMO_DIR.glob("*.png"))
                if demo_files:
                    local_path = demo_files[0]
                else:
                    raise HTTPException(status_code=404, detail="Original image not found for sensitivity analysis.")

    if isinstance(regions, str):
        try:
            regions = json.loads(regions)
        except Exception:
            regions = []

    res = screening_service.run_region_sensitivity(
        image_path=local_path,
        regions=regions,
        target_class=target_class
    )
    return JSONResponse(content=res)


@app.get("/api/download-report/{filename}")
async def download_report(filename: str):
    """Direct download endpoint for generated clinical reports (PDF and HTML)."""
    report_file = OUTPUTS_DIR / filename

    # If PDF is requested but doesn't exist yet, convert matching HTML report on-demand
    if filename.endswith(".pdf") and not report_file.exists():
        html_file = OUTPUTS_DIR / filename.replace(".pdf", ".html")
        if html_file.exists():
            try:
                from webapp.services.report_service import ReportService
                rep_service = ReportService(OUTPUTS_DIR)
                success = rep_service.convert_html_to_pdf(html_file, report_file)
                if not success or not report_file.exists():
                    raise HTTPException(status_code=500, detail="Failed to generate clinical PDF report.")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to generate clinical PDF report: {str(e)}")

    if not report_file.exists():
        raise HTTPException(status_code=404, detail="Report file not found.")

    if filename.endswith(".pdf"):
        with open(report_file, "rb") as f:
            header = f.read(5)
        if not header.startswith(b"%PDF-"):
            raise HTTPException(status_code=500, detail="Corrupted or invalid PDF report file.")
        return FileResponse(
            path=str(report_file),
            filename=filename,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )

    return FileResponse(
        path=str(report_file),
        filename=filename,
        media_type="text/html"
    )
