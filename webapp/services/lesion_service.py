"""
lesion_service.py
=================
Quantitative Retinal Lesion Candidate Analysis Service with AI Segmentation & Heuristic Fallback.

RetinaGuard 2.0 Compliance:
1. Reframe all findings strictly as candidate evidence:
   - Microaneurysm candidates
   - Hard exudate candidates
   - Hemorrhage candidates
   - Soft exudate candidates
2. Never call findings confirmed lesions unless independently validated.
3. Extract candidate crops with bounding boxes for transparent clinical inspection.
4. Heuristic Neovascularization Risk Indicator clearly labeled as a vessel density proxy.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import cv2

try:
    from python.inference.segment_lesions import LesionSegmentationPipeline
except ImportError:
    LesionSegmentationPipeline = None


class LesionService:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ai_pipeline = None

        if LesionSegmentationPipeline is not None:
            try:
                self.ai_pipeline = LesionSegmentationPipeline()
            except Exception as e:
                print(f"[LesionService] Note: AI segmentation model init notice ({e}). Fallback active.")

    def analyze(
        self,
        img_bgr: np.ndarray,
        od_mask: Optional[np.ndarray] = None,
        vessel_density: float = 10.0,
        filename_prefix: str = "lesions"
    ) -> Dict[str, Any]:
        # 1. Attempt AI Segmentation if pipeline is available and model is loaded
        if self.ai_pipeline is not None and self.ai_pipeline.is_ai_loaded:
            try:
                res = self._analyze_ai(img_bgr, vessel_density, filename_prefix)
                res["visual_crops"] = res.get("candidate_crops", [])
                return res
            except Exception as e:
                print(f"[LesionService] AI inference failed ({e}). Reverting to Heuristic Fallback.")

        # 2. Heuristic Fallback (Classical Computer Vision)
        res = self._analyze_heuristic(img_bgr, od_mask, vessel_density, filename_prefix)
        res["visual_crops"] = res.get("candidate_crops", [])
        return res

    # Alias for API compatibility
    detect = analyze

    def _extract_crops(
        self,
        img_bgr: np.ndarray,
        candidate_items: List[Dict[str, Any]],
        max_crops: int = 4
    ) -> List[Dict[str, Any]]:
        """Extracts high-resolution visual crops of candidate lesion regions."""
        h, w = img_bgr.shape[:2]
        crops = []
        for idx, item in enumerate(candidate_items[:max_crops]):
            bbox = item.get("bbox", [w // 2, h // 2, 20, 20])
            bx, by, bw, bh = bbox
            cx, cy = bx + bw // 2, by + bh // 2
            half_size = max(24, max(bw, bh) // 2 + 10)

            x0, y0 = max(0, cx - half_size), max(0, cy - half_size)
            x1, y1 = min(w, cx + half_size), min(h, cy + half_size)

            crop_img = img_bgr[y0:y1, x0:x1].copy()
            if crop_img.size == 0:
                continue

            # Standardize crop size to 96x96 for crisp UI display
            crop_resized = cv2.resize(crop_img, (96, 96), interpolation=cv2.INTER_LINEAR)
            # Add subtle corner bounding box
            cv2.rectangle(crop_resized, (2, 2), (93, 93), (0, 220, 255), 1)

            crop_filename = f"crop_{item['id']}_{int(np.random.randint(100000, 999999))}.png"
            crop_path = self.output_dir / crop_filename
            cv2.imwrite(str(crop_path), crop_resized)

            crops.append({
                "id": item["id"],
                "type": item["type"],
                "candidate_type": item["type"],
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "bounding_box": {"x": int(x0), "y": int(y0), "width": int(x1 - x0), "height": int(y1 - y0)},
                "area_px": int(item.get("area_px", bw * bh)),
                "crop_url": f"/outputs/{crop_filename}",
                "preview_url": f"/outputs/{crop_filename}",
                "description": f"{item['type']} at [{x0}, {y0}]",
                "detection_source": item.get("source", "Segmentation U-Net / Morphological Detector")
            })

        return crops

    def _analyze_ai(
        self,
        img_bgr: np.ndarray,
        vessel_density: float,
        filename_prefix: str
    ) -> Dict[str, Any]:
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ai_res = self.ai_pipeline.segment(img_rgb)

        # Compute Neovascularization Risk Indicator (measurable vessel characteristics)
        if vessel_density > 16.0:
            nv_risk = "High"
            nv_description = f"Vascular density elevated ({vessel_density:.1f}%). Possible active neovascular proliferation indicator."
        elif vessel_density > 13.5:
            nv_risk = "Moderate"
            nv_description = f"Vascular density borderline ({vessel_density:.1f}%). Monitor for microvascular remodeling."
        else:
            nv_risk = "Low"
            nv_description = f"Vascular density within expected screening range ({vessel_density:.1f}%)."

        # Save overlaid visualization
        overlay_rgb = ai_res["overlay_rgb"]
        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        overlay_filename = f"{filename_prefix}_overlay_{int(np.random.randint(100000, 999999))}.png"
        overlay_path = self.output_dir / overlay_filename
        cv2.imwrite(str(overlay_path), overlay_bgr)

        counts = ai_res["lesion_counts"]
        areas = ai_res["lesion_areas"]
        masks = ai_res.get("lesion_masks") or ai_res.get("masks", {})

        # Extract bounding boxes for candidate crops
        candidates_for_crops = []
        for l_type, mask in masks.items():
            if l_type == "OD":
                continue
            if mask is not None and np.any(mask):
                num_cc, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
                for i in range(1, min(num_cc, 3)):
                    bx = int(stats[i, cv2.CC_STAT_LEFT])
                    by = int(stats[i, cv2.CC_STAT_TOP])
                    bw = int(stats[i, cv2.CC_STAT_WIDTH])
                    bh = int(stats[i, cv2.CC_STAT_HEIGHT])
                    area = int(stats[i, cv2.CC_STAT_AREA])
                    type_labels = {
                        "MA": "Microaneurysm candidate",
                        "EX": "Hard exudate candidate",
                        "HE": "Hemorrhage candidate",
                        "SE": "Soft exudate candidate"
                    }
                    candidates_for_crops.append({
                        "id": f"{l_type.lower()}_{i}",
                        "type": type_labels.get(l_type, f"{l_type} candidate"),
                        "bbox": [bx, by, bw, bh],
                        "area_px": area,
                        "source": "Dual-Head U-Net ONNX (IDRiD expert ground truth trained)"
                    })

        crops = self._extract_crops(img_bgr, candidates_for_crops, max_crops=4)

        return {
            "mode": "AI SEGMENTATION",
            "badge_label": "CANDIDATE EVIDENCE (Dual-Head U-Net)",
            "is_ai": True,
            "disclaimer": "Candidate evidence regions identified by computer vision; not independently confirmed clinical lesions.",
            "microaneurysms": {
                "candidate_count": int(counts.get("MA", 0)),
                "total_area_px": int(areas.get("MA", 0)),
                "label": "Microaneurysm candidates",
                "methodology": "Dual-Head U-Net (IDRiD expert ground truth trained)"
            },
            "hard_exudates": {
                "candidate_count": int(counts.get("EX", 0)),
                "total_area_px": int(areas.get("EX", 0)),
                "label": "Hard exudate candidates",
                "methodology": "Dual-Head U-Net (IDRiD expert ground truth trained)"
            },
            "hemorrhages": {
                "candidate_count": int(counts.get("HE", 0)),
                "total_area_px": int(areas.get("HE", 0)),
                "label": "Hemorrhage candidates",
                "methodology": "Dual-Head U-Net (IDRiD expert ground truth trained)"
            },
            "soft_exudates": {
                "candidate_count": int(counts.get("SE", 0)),
                "total_area_px": int(areas.get("SE", 0)),
                "label": "Soft exudate candidates (Cotton wool spots)",
                "methodology": "Dual-Head U-Net (IDRiD expert ground truth trained)"
            },
            "neovascularization": {
                "risk_level": nv_risk,
                "label": "Neovascularization Risk Indicator",
                "density_heuristic": "Vessel density stratification heuristic indicator",
                "description": nv_description,
                "methodology": "Peri-papillary vessel density & branching stratification heuristic indicator (not clinical diagnosis)"
            },
            "candidate_crops": crops,
            "overlay_url": f"/outputs/{overlay_filename}",
            "legend": [
                {"name": "Microaneurysm candidates", "color": "#ff1744", "description": "Focal microvascular dilatations (MA)"},
                {"name": "Hemorrhage candidates", "color": "#ff9100", "description": "Intra-retinal blotches (HE)"},
                {"name": "Hard exudate candidates", "color": "#ffea00", "description": "Lipid deposits (EX)"},
                {"name": "Soft exudate candidates", "color": "#00e5ff", "description": "Cotton wool spots (SE)"},
                {"name": "Optic Disc", "color": "#00e676", "description": "Normal anatomical landmark (OD)"}
            ]
        }

    def _analyze_heuristic(
        self,
        img_bgr: np.ndarray,
        od_mask: Optional[np.ndarray] = None,
        vessel_density: float = 10.0,
        filename_prefix: str = "lesions"
    ) -> Dict[str, Any]:
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        green = img_bgr[:, :, 1]
        inv_green = 255 - green
        retina_mask = green > 15

        # 1. Optic Disc Exclusion Mask
        if od_mask is not None:
            kernel_disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            od_dilated = cv2.dilate(od_mask.astype(np.uint8), kernel_disc) > 0
        else:
            od_dilated = np.zeros((h, w), dtype=bool)

        # 2. Hard Exudate Candidate Segmentation
        ex_bright = (gray > 175) & (~od_dilated) & retina_mask
        num_ex, ex_labels, ex_stats, _ = cv2.connectedComponentsWithStats(ex_bright.astype(np.uint8))
        
        exudate_count = 0
        exudate_area = 0
        clean_ex_mask = np.zeros((h, w), dtype=bool)
        candidates_for_crops = []

        for i in range(1, num_ex):
            area = ex_stats[i, cv2.CC_STAT_AREA]
            if 5 <= area <= 4000:
                exudate_count += 1
                exudate_area += int(area)
                clean_ex_mask[ex_labels == i] = True
                if len(candidates_for_crops) < 2:
                    candidates_for_crops.append({
                        "id": f"ex_{exudate_count}",
                        "type": "Hard exudate candidate",
                        "bbox": [int(ex_stats[i, cv2.CC_STAT_LEFT]), int(ex_stats[i, cv2.CC_STAT_TOP]),
                                 int(ex_stats[i, cv2.CC_STAT_WIDTH]), int(ex_stats[i, cv2.CC_STAT_HEIGHT])],
                        "area_px": int(area),
                        "source": "High-luminance connected-component detector (Disc-excluded)"
                    })

        # 3. Microaneurysm Candidate Detection
        kernel_ma = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ma_tophat = cv2.morphologyEx(inv_green, cv2.MORPH_TOPHAT, kernel_ma)
        _, ma_thresh = cv2.threshold(ma_tophat, 22, 255, cv2.THRESH_BINARY)
        ma_candidates = (ma_thresh > 0) & retina_mask & (~od_dilated)
        
        num_ma, ma_labels, ma_stats, _ = cv2.connectedComponentsWithStats(ma_candidates.astype(np.uint8))
        ma_count = 0
        clean_ma_mask = np.zeros((h, w), dtype=bool)
        for i in range(1, num_ma):
            area = ma_stats[i, cv2.CC_STAT_AREA]
            if 3 <= area <= 65:
                ma_count += 1
                clean_ma_mask[ma_labels == i] = True
                if len(candidates_for_crops) < 3:
                    candidates_for_crops.append({
                        "id": f"ma_{ma_count}",
                        "type": "Microaneurysm candidate",
                        "bbox": [int(ma_stats[i, cv2.CC_STAT_LEFT]), int(ma_stats[i, cv2.CC_STAT_TOP]),
                                 int(ma_stats[i, cv2.CC_STAT_WIDTH]), int(ma_stats[i, cv2.CC_STAT_HEIGHT])],
                        "area_px": int(area),
                        "source": "Green-channel morphological top-hat detector"
                    })

        # 4. Intra-retinal Hemorrhage Candidate Detection
        he_candidates = (inv_green > 195) & (~od_dilated) & retina_mask
        num_he, he_labels, he_stats, _ = cv2.connectedComponentsWithStats(he_candidates.astype(np.uint8))
        he_count = 0
        he_area = 0
        clean_he_mask = np.zeros((h, w), dtype=bool)
        for i in range(1, num_he):
            area = he_stats[i, cv2.CC_STAT_AREA]
            if 40 <= area <= 2500:
                he_count += 1
                he_area += int(area)
                clean_he_mask[he_labels == i] = True
                if len(candidates_for_crops) < 4:
                    candidates_for_crops.append({
                        "id": f"he_{he_count}",
                        "type": "Hemorrhage candidate",
                        "bbox": [int(he_stats[i, cv2.CC_STAT_LEFT]), int(he_stats[i, cv2.CC_STAT_TOP]),
                                 int(he_stats[i, cv2.CC_STAT_WIDTH]), int(he_stats[i, cv2.CC_STAT_HEIGHT])],
                        "area_px": int(area),
                        "source": "Dark lesion intra-retinal component detector"
                    })

        # Compute Neovascularization Risk Indicator
        if vessel_density > 16.0:
            nv_risk = "High"
            nv_description = f"Vascular density elevated ({vessel_density:.1f}%). Possible active neovascular proliferation indicator."
        elif vessel_density > 13.5:
            nv_risk = "Moderate"
            nv_description = f"Vascular density borderline ({vessel_density:.1f}%). Monitor for vascular remodeling."
        else:
            nv_risk = "Low"
            nv_description = f"Vascular density within expected screening range ({vessel_density:.1f}%)."

        # Overlay generation
        overlay = img_bgr.copy()
        overlay[clean_ex_mask] = [0, 230, 255]
        ma_disp = cv2.dilate(clean_ma_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
        overlay[ma_disp] = [255, 0, 240]
        overlay[clean_he_mask] = [0, 0, 240]

        blended = cv2.addWeighted(img_bgr, 0.55, overlay, 0.45, 0)
        overlay_filename = f"{filename_prefix}_overlay_{int(np.random.randint(100000, 999999))}.png"
        overlay_path = self.output_dir / overlay_filename
        cv2.imwrite(str(overlay_path), blended)

        crops = self._extract_crops(img_bgr, candidates_for_crops, max_crops=4)

        return {
            "mode": "HEURISTIC FALLBACK",
            "badge_label": "CANDIDATE EVIDENCE (Classical Morphology)",
            "is_ai": False,
            "disclaimer": "Candidate evidence regions identified by computer vision; not independently confirmed clinical lesions.",
            "microaneurysms": {
                "candidate_count": int(ma_count),
                "total_area_px": int(np.sum(clean_ma_mask)),
                "label": "Microaneurysm candidates",
                "methodology": "Green-channel morphological top-hat filtering (3–65 px)"
            },
            "hard_exudates": {
                "candidate_count": int(exudate_count),
                "total_area_px": int(exudate_area),
                "label": "Hard exudate candidates",
                "methodology": "High-luminance intra-retinal lipid segmentation (Optic disc excluded)"
            },
            "hemorrhages": {
                "candidate_count": int(he_count),
                "total_area_px": int(he_area),
                "label": "Hemorrhage candidates",
                "methodology": "Dark lesion connected-component analysis (40–2500 px)"
            },
            "soft_exudates": {
                "candidate_count": 0,
                "total_area_px": 0,
                "label": "Soft exudate candidates (Not modeled in heuristic fallback)",
                "methodology": "N/A in heuristic mode"
            },
            "neovascularization": {
                "risk_level": nv_risk,
                "label": "Neovascularization Risk Indicator",
                "density_heuristic": "Vessel density stratification heuristic indicator",
                "description": nv_description,
                "methodology": "Peri-papillary vessel density & branching stratification heuristic indicator (not clinical diagnosis)"
            },
            "candidate_crops": crops,
            "overlay_url": f"/outputs/{overlay_filename}",
            "legend": [
                {"name": "Hard exudate candidates", "color": "#ffe600", "description": "Lipid deposits (Disc excluded)"},
                {"name": "Microaneurysm candidates", "color": "#ff00ea", "description": "Focal microvascular dilatations"},
                {"name": "Hemorrhage candidates", "color": "#ff2a2a", "description": "Intra-retinal blotches"}
            ]
        }
