"""
RetinaGuard — Processing & Filter Transparency Registry
========================================================
Maintains an exhaustive, auditable registry of every actual preprocessing,
filtering, anatomical, and machine learning stage in the RetinaGuard 2.0 pipeline.

Every stage explicitly documents:
- what_it_does
- why_it_is_used
- actual_parameters
- what_changed
- limitation

This registry powers the "Why?" transparency modals in the clinical workstation UI.
"""

from typing import Dict, Any, List


PIPELINE_STAGES: Dict[str, Dict[str, Any]] = {
    "stage_1_quality_gate": {
        "id": "stage_1_quality_gate",
        "name": "Image Quality Assessment Gate",
        "category": "Quality & Safety",
        "what_it_does": "Calculates objective optical metrics from the raw retinal photograph: Laplacian gradient variance (sharpness), mean green-channel intensity (illumination), standard deviation of active retinal pixels (contrast), and active sensor coverage (field of view).",
        "why_it_is_used": "To prevent clinical misdiagnosis caused by motion blur, underexposure, glare, or severe lens artifact. If an image is ungradable, downstream grading is halted immediately.",
        "actual_parameters": {
            "min_composite_score": 40.0,
            "borderline_score": 60.0,
            "sharpness_weight": 0.45,
            "illumination_weight": 0.25,
            "contrast_weight": 0.20,
            "fov_weight": 0.10,
            "min_laplacian_var": 35.0,
            "min_mean_lum": 30.0,
            "max_mean_lum": 225.0,
            "min_fov_percent": 25.0
        },
        "what_changed": "No pixel modification; outputs quality score, gradability status, and explicit recapture recommendation.",
        "limitation": "Configured prototype thresholds. Extreme peripheral pathology outside the central 45-degree field may not be detected by optical quality proxies alone."
    },
    "stage_2_border_removal": {
        "id": "stage_2_border_removal",
        "name": "Black Border & Non-Diagnostic Margin Removal",
        "category": "Preprocessing",
        "what_it_does": "Thresholds the fundus scan to segment the active circular retinal aperture from non-diagnostic black camera borders, computing a tight bounding box with 1% padding.",
        "why_it_is_used": "Fundus cameras produce varying black mask borders that skew neural network convolution spatial statistics and waste memory.",
        "actual_parameters": {
            "intensity_tolerance": 15,
            "padding_fraction": 0.01
        },
        "what_changed": "Crops outer zero-intensity margins so subsequent filters operate exclusively on retinal tissue.",
        "limitation": "Images with severe peripheral illumination falloff or crescent artifacts may suffer slight over-cropping at the border."
    },
    "stage_3_resolution_standardization": {
        "id": "stage_3_resolution_standardization",
        "name": "Geometric Resolution Standardization",
        "category": "Preprocessing",
        "what_it_does": "Resamples the cropped fundus to a standardized clinical workstation canvas (512x512 for landmark/lesion processing) and 224x224 for EfficientNet-B0 inference.",
        "why_it_is_used": "Ensures uniform scale for morphological filters (kernel sizes correspond to constant retinal anatomy) and matches the native receptive field of the neural backbone.",
        "actual_parameters": {
            "workstation_size": [512, 512],
            "classifier_input_size": [224, 224],
            "interpolation": "INTER_AREA (downsampling) / INTER_LINEAR"
        },
        "what_changed": "Scales the coordinate system so all downstream pixels map to known spatial proportions.",
        "limitation": "High-resolution 4K fundus images lose sub-pixel fine detail upon downsampling to 224x224 for classification."
    },
    "stage_4_green_channel": {
        "id": "stage_4_green_channel",
        "name": "Green-Channel Chromatic Separation",
        "category": "Filtering",
        "what_it_does": "Isolates the middle color channel (G) from the BGR/RGB photograph for structural and lesion analysis.",
        "why_it_is_used": "Ocular fundus hemoglobin absorption peaks in the green spectrum (~540-570 nm). The green channel exhibits the highest optical contrast between retinal microvasculature, hemorrhages, and the fundus background.",
        "actual_parameters": {
            "channel_index": 1,
            "color_space": "BGR -> Green (G)"
        },
        "what_changed": "Extracts single-channel 8-bit grayscale representation maximizing vessel and lesion visibility.",
        "limitation": "Choroidal vessels and deep melanin pigmentation visible in red channels are suppressed in green-channel extraction."
    },
    "stage_5_clahe": {
        "id": "stage_5_clahe",
        "name": "Contrast-Limited Adaptive Histogram Equalization (CLAHE)",
        "category": "Enhancement",
        "what_it_does": "Divides the image into an 8x8 grid of contextual tiles, computes local histograms, clips local contrast amplification to 2.2 to prevent noise over-amplification, and bilinearly interpolates across tile boundaries.",
        "why_it_is_used": "Retinal fundus images frequently suffer from non-uniform illumination (brighter at the optic disc, darker at peripheral edges). CLAHE balances local visibility of subtle microaneurysms and exudates across all quadrants.",
        "actual_parameters": {
            "clip_limit_green": 2.2,
            "clip_limit_mild_rb": 1.2,
            "tile_grid_size": [8, 8]
        },
        "what_changed": "Enhanced local contrast in under-illuminated quadrants while preventing glare saturation in bright peripapillary zones.",
        "limitation": "Can occasionally accentuate high-frequency background sensor noise in severely underexposed captures if not paired with denoising."
    },
    "stage_6_denoising": {
        "id": "stage_6_denoising",
        "name": "Micro-Scale Gaussian Denoising",
        "category": "Filtering",
        "what_it_does": "Applies a subtle 3x3 Gaussian spatial smoothing kernel with standard deviation sigma=0.5 to the recombined color image.",
        "why_it_is_used": "Smooths out single-pixel sensor noise generated by low-cost CMOS fundus sensors without blurring fine vessel boundaries.",
        "actual_parameters": {
            "kernel_size": [3, 3],
            "sigma": 0.5
        },
        "what_changed": "Slight attenuation of high-frequency noise spikes.",
        "limitation": "Very subtle tiny microaneurysms of diameter < 1 pixel could theoretically lose edge sharpness if excessive smoothing is applied."
    },
    "stage_7_anatomical_landmarks": {
        "id": "stage_7_anatomical_landmarks",
        "name": "Anatomical Landmark Localization & Vessel Density",
        "category": "Anatomy",
        "what_it_does": "Locates the Optic Disc center using Gaussian low-pass spatial peak detection and morphological radius estimation (8% min dimension); projects Fovea Centralis coordinates at 2.5 disc diameters temporal; extracts retinal vasculature using multi-scale Black Top-Hat morphology (11x11 and 5x5 kernels) and computes vascular density %.",
        "why_it_is_used": "Optic disc localization is essential to generate an exclusion mask so physiological brightness of the disc is not misclassified as hard exudates. Vessel density provides baseline vascular geometry.",
        "actual_parameters": {
            "od_blur_kernel": [31, 31],
            "od_radius_factor": 0.08,
            "fovea_temporal_offset_factor": 2.5,
            "vessel_tophat_large": [11, 11],
            "vessel_tophat_small": [5, 5],
            "vessel_weight_large": 0.6,
            "vessel_weight_small": 0.4
        },
        "what_changed": "Generates optic disc binary mask, fovea coordinates, vessel binary segmentation, and vascular overlay.",
        "limitation": "Geometric fovea projection assumes standard 45-degree macula-centered or disc-centered field. In highly tilted or non-standard fields, fovea position may have slight positional offset."
    },
    "stage_8_lesion_candidates": {
        "id": "stage_8_lesion_candidates",
        "name": "Candidate Lesion Evidence Detection",
        "category": "Lesion Evidence",
        "what_it_does": "Segments candidate microvascular abnormalities: Microaneurysm candidates, Hard Exudate candidates (with disc exclusion), and Hemorrhage candidates using Dual-Head U-Net ONNX with classical morphology fallback. Extracts candidate crops and bounding boxes.",
        "why_it_is_used": "Provides localized visual evidence explaining potential microvascular changes without making unverified clinical diagnostic claims.",
        "actual_parameters": {
            "model": "Dual-Head U-Net (ONNX) with Morphological Fallback",
            "exudate_brightness_threshold": 175,
            "hemorrhage_darkness_threshold": 195,
            "min_candidate_area_px": 3
        },
        "what_changed": "Generates labeled candidate lesion map, candidate counts, total pixel areas, and localized crops.",
        "limitation": "Candidate evidence only. These are candidate regions identified by computer vision and deep learning; they are not independently confirmed clinical lesion segmentations."
    },
    "stage_9_classifier_grading": {
        "id": "stage_9_classifier_grading",
        "name": "EXP-001 EfficientNet-B0 DR Severity Grading",
        "category": "Deep Learning",
        "what_it_does": "Executes forward inference using the trained EXP-001 model (ImageNet pretrained EfficientNet-B0 backbone fine-tuned with Focal Loss gamma=2.0 on APTOS 2019) via ONNX Runtime with PyTorch fallback. Computes complete 5-class softmax probability distribution.",
        "why_it_is_used": "Core screening classification providing standard ICDR 5-class DR severity prediction (Grade 0 to Grade 4).",
        "actual_parameters": {
            "model_architecture": "EfficientNet-B0",
            "loss_function": "Focal Loss (gamma=2.0)",
            "num_classes": 5,
            "runtime_engine": "ONNX Runtime (CPU) with PyTorch fallback",
            "input_resolution": [224, 224],
            "referral_threshold_grade": 2
        },
        "what_changed": "Produces predicted grade, probability distribution vector, and referral screening triage.",
        "limitation": "Model probabilities are raw softmax outputs without post-hoc Platt scaling/isotonic calibration (Calibration: not applied). Research screening prototype."
    },
    "stage_10_explainability": {
        "id": "stage_10_explainability",
        "name": "Model Attribution Explainability (Grad-CAM++)",
        "category": "XAI Attribution",
        "what_it_does": "Calculates gradient-weighted class activation maps using 2nd and 3rd order gradients from the final Conv2d layer (features.8[0]) w.r.t. the predicted (or user-selected) class. Validates map integrity; falls back to standard Grad-CAM if needed.",
        "why_it_is_used": "Exposes spatial regions of the fundus that contributed positively to the neural network's decision. Never substitutes synthetic/fake heatmaps.",
        "actual_parameters": {
            "primary_method": "Grad-CAM++",
            "fallback_method": "Grad-CAM",
            "target_layer": "backbone.features.8.0 (final Conv2d, 1280 channels)",
            "blend_alpha": 0.45,
            "colormap": "cv2.COLORMAP_JET"
        },
        "what_changed": "Generates visual attribution overlay highlighting influential image regions.",
        "limitation": "Attribution visualization shows regions contributing to model prediction; it is not confirmed lesion segmentation or proof of causality."
    },
    "stage_11_screening_interpretation": {
        "id": "stage_11_screening_interpretation",
        "name": "Clinical Screening Interpretation",
        "category": "Triage",
        "what_it_does": "Evaluates predicted grade against the configured prototype screening rule (Grade >= 2 = Referable DR) and generates actionable follow-up advice.",
        "why_it_is_used": "Translates quantitative model predictions into structured triage recommendations for community healthcare screeners.",
        "actual_parameters": {
            "rule": "Grade >= 2 -> Refer to ophthalmologist",
            "grade_0": "Routine follow-up in 12 months",
            "grade_1": "Consider follow-up in 6-12 months",
            "grade_2": "Ophthalmology evaluation recommended in 3-6 months",
            "grade_3": "Ophthalmology referral required within 1 month",
            "grade_4": "Urgent ophthalmology referral required"
        },
        "what_changed": "Generates screening triage badge and timeline.",
        "limitation": "Configured prototype rule. Does not constitute autonomous diagnosis or replace clinical judgment."
    },
    "stage_12_reporting": {
        "id": "stage_12_reporting",
        "name": "Clinical Screening Report Generation",
        "category": "Audit & Export",
        "what_it_does": "Compiles the single shared AnalysisResult into a self-contained, printable HTML clinical summary including quality metrics, DR severity, candidate evidence, attribution overlays, grounded explanations, and disclaimers.",
        "why_it_is_used": "Provides permanent, auditable, offline documentation for referral workflows.",
        "actual_parameters": {
            "format": "Standalone HTML5 / CSS print sheet",
            "source_of_truth": "Unified AnalysisResult object"
        },
        "what_changed": "Saves timestamped HTML file to outputs directory.",
        "limitation": "Static record of AI-assisted screening session."
    }
}


def get_all_stages() -> List[Dict[str, Any]]:
    """Returns the list of all registered pipeline stages with dual alias support."""
    stages = []
    for k, v in PIPELINE_STAGES.items():
        item = dict(v)
        item["stage_id"] = k
        item["why_used"] = v.get("why_it_is_used", "")
        item["what_changed_in_image"] = v.get("what_changed", "")
        stages.append(item)
    return stages


def get_stage_info(stage_id: str) -> Dict[str, Any]:
    """Returns detailed transparency metadata for a specific pipeline stage with dual aliases."""
    raw = PIPELINE_STAGES.get(stage_id)
    if not raw:
        return {
            "id": stage_id,
            "stage_id": stage_id,
            "name": "Unknown Stage",
            "what_it_does": "Not documented in transparency registry.",
            "why_used": "N/A",
            "why_it_is_used": "N/A",
            "actual_parameters": {},
            "what_changed": "N/A",
            "what_changed_in_image": "N/A",
            "limitation": "Unregistered stage."
        }
    item = dict(raw)
    item["stage_id"] = stage_id
    item["why_used"] = raw.get("why_it_is_used", "")
    item["what_changed_in_image"] = raw.get("what_changed", "")
    return item


class PipelineRegistry:
    """Namespace providing class-level access to the Pipeline Transparency Registry."""
    STAGES = PIPELINE_STAGES

    @classmethod
    def get_all_stages(cls) -> List[Dict[str, Any]]:
        return get_all_stages()

    @classmethod
    def get_stage_info(cls, stage_id: str) -> Dict[str, Any]:
        return get_stage_info(stage_id)


