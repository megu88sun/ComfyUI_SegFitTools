# ==============================================================================
# File Name:    mask_cleanup.py
# Description:  Applies reusable cleanup operations to an arbitrary mask.
# Proposer:     MeguPaPa3
# Author:       OpenAI Codex
# Created:      2026-07-18
#
# Change Log:
# Date        Author        Description
# ----------  ------------  --------------------------------------------------
# 2026-07-18  OpenAI Codex  Initial release.
# ==============================================================================
"""Reusable cleanup for masks produced by SegFormer, SAM, or manual tools."""

from __future__ import annotations

import torch

from .mask_refiner import SegFormerMaskRefiner


class MaskCleanup:
    """Apply the Refiner cleanup stage to any ComfyUI MASK."""

    CATEGORY = "SegFitTools/Mask"
    FUNCTION = "cleanup"
    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("cleaned_mask", "mask_preview")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "remove_islands_smaller_than": (
                    "INT", {"default": 80, "min": 0, "max": 1000000}
                ),
                "fill_holes_smaller_than": (
                    "INT", {"default": 80, "min": 0, "max": 1000000}
                ),
                "grow_shrink_pixels": (
                    "INT", {"default": 0, "min": -128, "max": 128}
                ),
                "feather_pixels": (
                    "FLOAT", {"default": 1.0, "min": 0.0, "max": 64.0, "step": 0.1}
                ),
                "debug_mode": (["standard", "detailed"], {"default": "standard"}),
            },
            "optional": {
                "profile": ("SEGMASK_PROFILE",),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, mask=None, **_kwargs):
        if mask is not None and (
            not isinstance(mask, torch.Tensor) or mask.ndim not in (2, 3)
        ):
            return "[Mask Cleanup] mask must be a MASK tensor (H,W) or (B,H,W)."
        return True

    def cleanup(self, mask, remove_islands_smaller_than,
                fill_holes_smaller_than, grow_shrink_pixels, feather_pixels,
                debug_mode, profile=None):
        try:
            from scipy import ndimage

            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim != 3:
                raise ValueError("mask must have shape (height, width) or (batch, height, width).")
            if not torch.isfinite(mask).all():
                raise ValueError("mask contains NaN or infinite values.")

            fill_holes_smaller_than, feather_pixels = (
                SegFormerMaskRefiner._profile_cleanup(
                    profile, int(fill_holes_smaller_than), float(feather_pixels)
                )
            )
            binary = mask.clamp(0, 1) > 0.5
            cleaned = SegFormerMaskRefiner()._clean(
                binary,
                remove_islands_smaller_than,
                fill_holes_smaller_than,
                grow_shrink_pixels,
                feather_pixels,
                ndimage,
            )
            preview = cleaned.unsqueeze(-1).repeat(1, 1, 1, 3)
            return cleaned, preview
        except Exception as exc:
            if debug_mode == "detailed":
                raise
            raise RuntimeError(f"[Mask Cleanup] Cleanup failed: {exc}") from None
