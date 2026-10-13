# ==============================================================================
# File Name:    mask_auto_switching.py
# Description:  Selects the single active MASK from mutually exclusive branches.
# Proposer:     MeguPaPa3
# Author:       OpenAI Codex
# Created:      2026-07-19
#
# Change Log:
# Date        Author        Description
# ----------  ------------  --------------------------------------------------
# 2026-07-19  OpenAI Codex  Initial release.
# ==============================================================================
"""Exclusive automatic MASK switch for mutually muted workflow branches."""

from __future__ import annotations

import torch


class MaskAutoSwitching:
    """Pass the only available MASK input and reject zero or multiple inputs."""

    CATEGORY = "SegFitTools/Switch"
    FUNCTION = "switch"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    MAX_INPUTS = 8

    @classmethod
    def INPUT_TYPES(cls):
        optional = {f"mask_{index}": ("MASK",) for index in range(1, cls.MAX_INPUTS + 1)}
        return {
            "required": {
                "input_count": (
                    "INT", {"default": 2, "min": 2, "max": cls.MAX_INPUTS, "step": 1}
                ),
            },
            "optional": optional,
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_count=2, **kwargs):
        count = int(input_count)
        if count < 2 or count > cls.MAX_INPUTS:
            return f"[Mask Auto Switching] input_count must be between 2 and {cls.MAX_INPUTS}."
        for index in range(1, count + 1):
            value = kwargs.get(f"mask_{index}")
            if value is not None and (
                not isinstance(value, torch.Tensor) or value.ndim not in (2, 3)
            ):
                return f"[Mask Auto Switching] mask_{index} must be a MASK tensor."
        return True

    @staticmethod
    def switch(input_count=2, **kwargs):
        count = int(input_count)
        active = [
            (index, kwargs.get(f"mask_{index}"))
            for index in range(1, count + 1)
            if kwargs.get(f"mask_{index}") is not None
        ]
        if not active:
            raise RuntimeError(
                "[Mask Auto Switching] No mask input is active. Enable exactly one branch."
            )
        if len(active) > 1:
            names = ", ".join(f"mask_{index}" for index, _value in active)
            raise RuntimeError(
                f"[Mask Auto Switching] Multiple mask inputs are active ({names}). "
                "Mute all but one branch."
            )
        return (active[0][1],)
