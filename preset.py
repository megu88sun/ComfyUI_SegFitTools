# ==============================================================================
# File Name:    preset.py
# Description:  Provides lingerie-aware mask cleanup presets.
# Proposer:     MeguPaPa3
# Author:       OpenAI Codex
# Created:      2026-07-15
#
# Change Log:
# Date        Author        Description
# ----------  ------------  --------------------------------------------------
# 2026-07-15  OpenAI Codex  Initial release.
# ==============================================================================
"""Preset data for predictable lingerie-mask cleanup."""


class LingerieMaskPreset:
    CATEGORY = "SegFitTools/Mask"
    FUNCTION = "build"
    RETURN_TYPES = ("SEGMASK_PROFILE",)
    RETURN_NAMES = ("profile",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "colour_profile": (
                    ["white_or_very_light", "light_warm", "light_cool", "medium_or_dark"],
                    {"default": "white_or_very_light"},
                ),
                "lace_amount": (["none_or_minimal", "partial", "full"], {"default": "partial"}),
                "visual_opacity": (
                    ["opaque_or_lined", "semi_sheer", "sheer"],
                    {"default": "opaque_or_lined"},
                ),
                "lining": (["none", "partial", "full"], {"default": "partial"}),
            }
        }

    def build(self, colour_profile, lace_amount, visual_opacity, lining):
        return ({
            "colour_profile": colour_profile,
            "lace_amount": lace_amount,
            "visual_opacity": visual_opacity,
            "lining": lining,
        },)

