# ==============================================================================
# File Name:    detected_mask_switching.py
# Description:  Lazily selects Restorer or Poly Cutter full-canvas mask output.
# Proposer:     MeguPaPa3
# Author:       OpenAI Codex
# Created:      2026-10-08
#
# Change Log:
# Date        Author        Description
# ----------  ------------  --------------------------------------------------
# 2026-10-08  OpenAI Codex  Initial release.
# 2026-10-08  MeguPaPa3     The order of the input links has been changed.
# ==============================================================================

from __future__ import annotations

import torch


class DetectedMaskSwitching:
    CATEGORY = "SegFitTools/Switch"
    FUNCTION = "switch"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    ROUTES = {"RESTORER": "A", "POLY_CUTTER": "B"}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "A": ("MASK", {"lazy": True}),
                "B": ("MASK", {"lazy": True}),
            },
            "required": {"route": ("STRING", {"forceInput": True})},
        }

    @classmethod
    def VALIDATE_INPUTS(cls, route=None, A=None, B=None):
        if route is not None and route not in cls.ROUTES:
            return "[Detected Mask Switching] route must be RESTORER or POLY_CUTTER."
        for name, value in (("A", A), ("B", B)):
            if value is not None and (
                not isinstance(value, torch.Tensor) or value.ndim not in (2, 3)
            ):
                return f"[Detected Mask Switching] {name} must be a MASK tensor."
        return True

    @classmethod
    def check_lazy_status(cls, route, A=None, B=None):
        if route not in cls.ROUTES:
            return []
        name = cls.ROUTES[route]
        value = A if name == "A" else B
        return [name] if value is None else []

    @classmethod
    def switch(cls, route, A=None, B=None):
        if route not in cls.ROUTES:
            raise RuntimeError(
                "[Detected Mask Switching] Unknown route."
            )
        name = cls.ROUTES[route]
        value = A if name == "A" else B
        if value is None:
            raise RuntimeError(
                f"[Detected Mask Switching] Route {route} produced no mask."
            )
        return (value,)
