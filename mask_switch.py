# ==============================================================================
# File Name:    mask_switch.py
# Description:  Provides a manual two-input MASK passthrough switch.
# Proposer:     MeguPaPa3
# Author:       OpenAI Codex
# Created:      2026-07-18
#
# Change Log:
# Date        Author        Description
# ----------  ------------  --------------------------------------------------
# 2026-07-18  OpenAI Codex  Initial release.
# ==============================================================================
"""Manual two-input MASK passthrough switch."""

from __future__ import annotations

import torch


class SwitchMask:
    """Pass either mask_1 or mask_2 through without modifying it."""

    CATEGORY = "SegFitTools/Mask"
    FUNCTION = "select_mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_1": ("MASK",),
                "mask_2": ("MASK",),
                "select": (["mask_1", "mask_2"], {"default": "mask_1"}),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, mask_1=None, mask_2=None, select=None, **_kwargs):
        for name, value in (("mask_1", mask_1), ("mask_2", mask_2)):
            if value is not None and (
                not isinstance(value, torch.Tensor) or value.ndim not in (2, 3)
            ):
                return f"[Switch Mask] {name} must be a MASK tensor (H,W) or (B,H,W)."
        if select is not None and select not in ("mask_1", "mask_2"):
            return "[Switch Mask] select must be mask_1 or mask_2."
        return True

    @staticmethod
    def select_mask(mask_1, mask_2, select):
        return (mask_1 if select == "mask_1" else mask_2,)
