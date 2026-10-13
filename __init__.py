# ==============================================================================
# File Name:    __init__.py
# Description:  Registers the SegFitTools ComfyUI node collection.
# Proposer:     MeguPaPa3
# Author:       OpenAI Codex
# Created:      2026-07-15
#
# Change Log:
# Date        Author        Description
# ----------  ------------  --------------------------------------------------
# 2026-07-15  OpenAI Codex  Initial release.
# 2026-07-18  OpenAI Codex  Added mask refinement, cleanup, and switching nodes.
# 2026-07-19  OpenAI Codex  Added exclusive automatic image and mask switching nodes.
# 2026-07-19  OpenAI Codex  Added solid-color replacement for masked image areas.
# 2026-07-21  OpenAI Codex  Added existing-input image loading without an upload control.
# 2026-07-21  OpenAI Codex  Added source-based mask saving and conflict handling.
# 2026-07-24  OpenAI Codex  Added chained path image loading with alpha preservation.
# 2026-07-24  MeguPaPa3     Added Composite the source image by using a grayscale version of its mask image.
# 2026-07-26  OpenAI Codex  Added source-named temporary image saving.
# 2026-07-27  OpenAI Codex  Added a low-profile image loader.
# 2026-07-30  MeguPaPa3     Changed save nodes to Save As Image and Save As Mask.
# 2026-08-29  OpenAI Codex  Added scale-preserving lingerie photo splitting.
# 2026-09-24  Google Gemini Added DensePose part extraction and color-based masking.
# 2026-09-28  OpenAI Codex  Archived 001/002 investigation-only TEST nodes.
# 2026-10-01  OpenAI Codex  Updated Body Line Detector with R31 mask output.
# 2026-10-01  OpenAI Codex  Updated Body Line Detector with R32 outline validation.
# 2026-10-01  OpenAI Codex  Updated Body Line Detector with R33 local recovery.
# 2026-10-01  OpenAI Codex  Added dedicated NumPy temporary save/load nodes.
# 2026-10-01  OpenAI Codex  Updated Body Line Detector with R34 shared closure.
# 2026-10-02  OpenAI Codex  Updated Body Line Detector with R35 geometry checks.
# 2026-10-02  OpenAI Codex  Added R35.1 single-output HED centerline debug mode.
# 2026-10-02  OpenAI Codex  Added R35.2 Canny-validated SUPPORT gap recovery.
# 2026-10-02  OpenAI Codex  Added R35.3 Canny-following bridge routing.
# 2026-10-02  OpenAI Codex  Added R36 removal-envelope mask generation.
# 2026-10-02  OpenAI Codex  Promoted DensePose cropping and added mask restore.
# 2026-10-03  OpenAI Codex  Integrated DensePose location and evidence cropping.
# 2026-10-04  OpenAI Codex  Added R37 upper/lower Body Line Detector selection.
# 2026-10-04  OpenAI Codex  Aligned R37.1 Detector inputs with Cropper outputs.
# 2026-10-04  OpenAI Codex  Added R37.2 JSON-based garment-region inference.
# 2026-10-04  OpenAI Codex  Added separate R38 line-first detector experiment.
# 2026-10-04  OpenAI Codex  Revised R38 with boundary-origin line ownership.
# 2026-10-05  OpenAI Codex  Added separate HED-only R375 detector experiment.
# 2026-10-05  OpenAI Codex  Added coordinate-preserving HED/Canny overlay.
# 2026-10-06  OpenAI Codex  Added temporary loaded-photo JSON crop/save node.
# 2026-10-06  OpenAI Codex  Label R375 optional Canny-assisted local refinement.
# 2026-10-06  OpenAI Codex  Added separate exterior-space R376 detector.
# 2026-10-07  OpenAI Codex  Documented R376's appended Unresolved debug output;
#                           node registration remains unchanged.
# 2026-10-07  OpenAI Codex  R376 TY-16 keeps the same registered node mapping.
# 2026-10-08  OpenAI Codex  R376 TY-17 keeps the same sockets and separates
#                           open crop-bound Body detours from valid Shared.
# 2026-10-08  OpenAI Codex  R376 TY-18 keeps the same sockets and reconsiders
#                           narrow Lower faces with bilateral waist routes.
# 2026-10-08  OpenAI Codex  R376 TY-19 keeps the same sockets and separates
#                           narrow-face crotch and inner-pattern routes.
# 2026-10-08  OpenAI Codex  R376 TY-20 keeps the same sockets and layers open
#                           Upper side-band intervals onto Body as Shared.
# 2026-10-08  OpenAI Codex  Promoted R376 beta to Body Line Detector, archived
#                           the registered revision nodes, and appended Option.
# 2026-10-08  OpenAI Codex  Added no-value Restorer routing and locked-mask Poly
#                           Cutter fallback for incomplete detector pairs.
# 2026-10-08  OpenAI Codex  Added authoritative Restorer routing and lazy A/B
#                           selection without Poly Cutter-to-Restorer wiring.
# 2026-10-08  OpenAI Codex  Split route-aware Detected Mask Switching from the
#                           original multi-input Mask Auto Switching node.
# 2026-10-09  OpenAI Codex  Renamed DensePose Body Cropper to Body Line Cropper
#                           and reordered its upper/lower evidence outputs.
# 2026-10-09  MeguPaPa3     Duplicated Body Line Detector as Underwear Detector.
# 2026-10-09  OpenAI Codex  Fixed Underwear Detector registration and retained
#                           disabled legacy analysis/test node registrations.
# 2026-10-10  OpenAI Codex  Added Body Line crop-aware mannequin scaling.
# 2026-10-10  OpenAI Codex  Added DensePose-only Mannequin Cropper.
# 2026-10-10  OpenAI Codex  Added alpha compositing to Mannequin Body Scaler.
# 2026-10-13  OpenAI Codex  Added DensePose-only crop-data output.
# 2026-10-13  OpenAI Codex  Simplified Split Lingerie Photo to automatic
#                           full-mask separation and transparent garment crops.
# 2026-10-13  OpenAI Codex  Added optional premium overlay registration.
# ==============================================================================


# ==============================================================================
# Production Nodes
# ==============================================================================

from .garment_fitter import FitGarmentToTargetMask
from .advanced_fitter import FitGarmentAdvanced, FitReviewGate
from .mask_refiner import SegFormerMaskRefiner
from .mask_cleanup import MaskCleanup
from .mask_switch import SwitchMask
from .image_auto_switching import ImageAutoSwitching
from .mask_auto_switching import MaskAutoSwitching
from .detected_mask_switching import DetectedMaskSwitching
from .masked_area_fill import MaskedAreaFill
from .load_image_no_upload import LoadImageNoUpload
from .save_as_mask import SaveAsMask
from .load_image_by_chain import LoadImageByChain
from .preset import LingerieMaskPreset
from .pro_fitter import FitGarmentPro
from .join_maskedgray import JoinMaskedAreaGray
from .save_as_image import SaveAsImage
from .load_image_no_upload_lm import LoadImageNoUploadLM
from .lingerie_splitter import SplitLingeriePhoto
from .masked_area_skin_fill import MaskedAreaSkinFill
from .lingerie_boundary_enhancer import LingerieBoundaryEnhancer
from .garment_coarse_extractor import GarmentCoarseExtractor
from .garment_poly_cutter import GarmentPolyCutter
from .base_canvas_prepper import BaseCanvasPrepper
from .save_as_tmp import SaveAsTmp
from .load_tmp import LoadTmp
from .densepose_cropper import DensePoseCropper
from .underwear_mask_restorer import UnderwearMaskRestorer
from .mannequin_cropper import MannequinCropper
from .mannequin_body_scaler import MannequinBodyScaler


# ==============================================================================
# SegFit Analysis Base Nodes
#
# Keep only the analysis nodes that may still be used as the input/base for
# the 003 BODY-line exclusion stage.
# ==============================================================================

# These legacy analysis nodes are intentionally not imported or registered.
# from .lineart_canny_relation import LineArtCannyRelation
# from .hed_relation import HEDRelation
# from .hed_canny_overlay import HEDCannyOverlay


# ==============================================================================
# SegFit TEST Infrastructure
# ==============================================================================

from .save_segfit_test_checkpoint import SaveSegFitTestCheckpoint
from .load_segfit_test_checkpoint import LoadSegFitTestCheckpoint
from .debug_body_crop_save import DebugBodyCropSave  # Temporary debug registration.

# These investigation-only nodes are intentionally disabled in production.
# from .densepose_part_extractor import DensePosePartExtractor
# from .garment_anchor_detector import GarmentAnchorDetector
# from .garment_line_detector_node import GarmentLineDetectorNode


# ==============================================================================
# Web
# ==============================================================================

WEB_DIRECTORY = "./web"


# ==============================================================================
# Node Class Mappings
# ==============================================================================

NODE_CLASS_MAPPINGS = {

    # --------------------------------------------------------------------------
    # Production
    # --------------------------------------------------------------------------

    "SegFormerMaskRefiner": SegFormerMaskRefiner,
    "MaskCleanup": MaskCleanup,
    "SwitchMask": SwitchMask,
    "ImageAutoSwitching": ImageAutoSwitching,
    "MaskAutoSwitching": MaskAutoSwitching,
    "DetectedMaskSwitching": DetectedMaskSwitching,
    "MaskedAreaFill": MaskedAreaFill,
    "LoadImageNoUpload": LoadImageNoUpload,
    "SaveAsMask": SaveAsMask,
    "LoadImageByChain": LoadImageByChain,
    "LingerieMaskPreset": LingerieMaskPreset,
    "FitGarmentToTargetMask": FitGarmentToTargetMask,
    "FitGarmentAdvanced": FitGarmentAdvanced,
    "FitReviewGate": FitReviewGate,
    "FitGarmentPro": FitGarmentPro,
    "JoinMaskedAreaGray": JoinMaskedAreaGray,
    "SaveAsImage": SaveAsImage,
    "LoadImageNoUploadLM": LoadImageNoUploadLM,
    "SplitLingeriePhoto": SplitLingeriePhoto,
    "MaskedAreaSkinFill": MaskedAreaSkinFill,
    "LingerieBoundaryEnhancer": LingerieBoundaryEnhancer,
    "GarmentCoarseExtractor": GarmentCoarseExtractor,
    "GarmentPolyCutter": GarmentPolyCutter,
    "BaseCanvasPrepper": BaseCanvasPrepper,
    "SaveAsTmp": SaveAsTmp,
    "LoadTmp": LoadTmp,
    "DensePoseCropper": DensePoseCropper,
    "UnderwearMaskRestorer": UnderwearMaskRestorer,
    "MannequinCropper": MannequinCropper,
    "MannequinBodyScaler": MannequinBodyScaler,

    # --------------------------------------------------------------------------
    # Analysis Base
    # --------------------------------------------------------------------------

    # "LineArtCannyRelation": LineArtCannyRelation,
    # "HEDRelation": HEDRelation,
    # "HEDCannyOverlay": HEDCannyOverlay,

    # --------------------------------------------------------------------------
    # TEST Infrastructure
    # --------------------------------------------------------------------------

    "SaveSegFitTestCheckpoint": SaveSegFitTestCheckpoint,
    "LoadSegFitTestCheckpoint": LoadSegFitTestCheckpoint,
    "DebugBodyCropSave": DebugBodyCropSave,  # Temporary debug registration.

    # "DensePosePartExtractor": DensePosePartExtractor,
    # "GarmentAnchorDetector": GarmentAnchorDetector,
    # "GarmentLineDetectorNode": GarmentLineDetectorNode,
}


# ==============================================================================
# Node Display Name Mappings
# ==============================================================================

NODE_DISPLAY_NAME_MAPPINGS = {

    # --------------------------------------------------------------------------
    # Production
    # --------------------------------------------------------------------------

    "SegFormerMaskRefiner": "SegFormer Mask Refiner",
    "MaskCleanup": "Mask Cleanup",
    "SwitchMask": "Switch Mask",
    "ImageAutoSwitching": "Image Auto Switching",
    "MaskAutoSwitching": "Mask Auto Switching",
    "DetectedMaskSwitching": "Detected Mask Switching",
    "MaskedAreaFill": "Masked Area Fill",
    "LoadImageNoUpload": "Load Image No Upload",
    "SaveAsMask": "Save As Mask",
    "LoadImageByChain": "Load Image By Chain",
    "LingerieMaskPreset": "Lingerie Mask Preset",
    "FitGarmentToTargetMask": "Fit Garment to Target Mask",
    "FitGarmentAdvanced": "Fit Garment Advanced",
    "FitReviewGate": "Fit Review Gate",
    "FitGarmentPro": "Fit Garment Pro",
    "JoinMaskedAreaGray": "Join Masked Area Gray",
    "SaveAsImage": "Save As Image",
    "LoadImageNoUploadLM": "Load Image No Upload LM",
    "SplitLingeriePhoto": "Split Lingerie Photo",
    "MaskedAreaSkinFill": "Masked Area Skin Fill",
    "LingerieBoundaryEnhancer": "Lingerie Boundary Enhancer",
    "GarmentCoarseExtractor": "Garment Coarse Extractor",
    "GarmentPolyCutter": "Garment Poly Cutter",
    "BaseCanvasPrepper": "Base Canvas Prepper",
    "SaveAsTmp": "Save As Tmp (SegFit)",
    "LoadTmp": "Load Tmp (SegFit)",
    "DensePoseCropper": "DensePose Cropper",
    "UnderwearMaskRestorer": "Underwear Mask Restorer",
    "MannequinCropper": "Mannequin Cropper",
    "MannequinBodyScaler": "Mannequin Body Scaler",

    # --------------------------------------------------------------------------
    # Analysis Base
    # --------------------------------------------------------------------------

    # "LineArtCannyRelation": "LineArt-Canny Relation (TEST)",
    # "HEDRelation": "HED Relation (TEST)",
    # "HEDCannyOverlay": "HED Canny Overlay (TEST)",

    # --------------------------------------------------------------------------
    # TEST Infrastructure
    # --------------------------------------------------------------------------

    "SaveSegFitTestCheckpoint": "Save SegFit TEST Checkpoint",
    "LoadSegFitTestCheckpoint": "Load SegFit TEST Checkpoint",
    # "DensePosePartExtractor": "Denspose Part Extractor(TEST)",
    # "GarmentAnchorDetector": "Garment Anchor Detector (TEST)",
    # "GarmentLineDetectorNode": "Garment Line Detector (TEST)",
    #
    # Please feel free to use this for your tests as well.
    "DebugBodyCropSave": "Body Crop and Save (DEBUG)",
}


# ==============================================================================
# Optional Premium Overlay
# ==============================================================================

# The public package omits premium_registry.py and premium implementation files.
# Copying the premium archive over the same folder adds these mappings without
# changing the ComfyUI node-pack directory name.
try:
    from .premium_registry import (
        PREMIUM_NODE_CLASS_MAPPINGS,
        PREMIUM_NODE_DISPLAY_NAME_MAPPINGS,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {f"{__package__}.premium_registry", "premium_registry"}:
        raise
    PREMIUM_NODE_CLASS_MAPPINGS = {}
    PREMIUM_NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_CLASS_MAPPINGS.update(PREMIUM_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(PREMIUM_NODE_DISPLAY_NAME_MAPPINGS)


# ==============================================================================
# Exports
# ==============================================================================

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
