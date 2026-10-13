# ==============================================================================
# File Name   : base_canvas_prepper.py
# Description : Prepares the raw mannequin/base image conditioning before any
#               garment try-on stage by unifying surface tone and prompt flow.
# Proposer    : MeguPaPa3
# Author      : Google Gemini
# Created     : 2026-09-18
#
# Change Log  :
# Date        Author         Description
# ----------  -------------  ----------------------------------------------------
# 2026-09-18  Google Gemini  Renamed from garment conditioner to base prepper
#                            to avoid confusion with the try-on fitting pipeline.
# 2026-09-21  Google Gemini  Fixed comfy imports and added fallback support
#                            for standalone testing environments.
# ==============================================================================

from __future__ import annotations

import os
from typing import Any

try:
    import comfy.sd
    import comfy.utils
    import folder_paths
except ImportError:
    # Standalone/pytest fallback when running outside the full ComfyUI runtime
    import sys
    from unittest.mock import MagicMock

    comfy = MagicMock()
    comfy.sd = MagicMock()
    comfy.utils = MagicMock()

    folder_paths = MagicMock()

    sys.modules.setdefault(
        "comfy",
        comfy,
    )

    sys.modules.setdefault(
        "comfy.sd",
        comfy.sd,
    )

    sys.modules.setdefault(
        "comfy.utils",
        comfy.utils,
    )

    sys.modules.setdefault(
        "folder_paths",
        folder_paths,
    )

class BaseCanvasPrepper:
    """
    Base canvas preparation node designed strictly for the initial mannequin stage.
    Applies an internal surface-smoothing patch and base prompt triggers prior to
    any garment fitting or try-on operations.
    """

    CATEGORY = "SegFitTools/BasePrep"
    FUNCTION = "prepare_base_canvas"
    RETURN_TYPES = ("MODEL", "CLIP", "CONDITIONING", "STRING")
    RETURN_NAMES = ("model", "clip", "positive_conditioning", "merged_prompt_text")

    # Internal preset LoRA filename stored under 'models/loras/'
    PRESET_LORA_NAME = "surface_tape_preset.safetensors"
    # Default trigger words intended solely for neutral under-layer canvas setup
    DEFAULT_TRIGGER_WORDS = "skin tone cover, smooth mannequin surface, modesty flat tape"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "base_positive_text": (
                    "STRING",
                    {
                        "default": "high quality, realistic body contour",
                        "multiline": True,
                        "tooltip": "Positive prompt for the base mannequin or figure body prior to fitting.",
                    },
                ),
                "prep_intensity": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Intensity of the internal canvas-smoothing surface patch.",
                    },
                ),
                "enable_prep": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Toggle whether to apply base-prep triggers and weight adjustments.",
                    },
                ),
            },
            "optional": {
                "extra_keywords": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Supplemental keywords appended specifically to the base preparation stage.",
                    },
                ),
            },
        }

    def prepare_base_canvas(
        self,
        model: Any,
        clip: Any,
        base_positive_text: str,
        prep_intensity: float,
        enable_prep: bool,
        extra_keywords: str = "",
    ) -> tuple[Any, Any, Any, str]:
        patched_model = model
        patched_clip = clip
        final_prompt = base_positive_text.strip()

        # Apply only if prep toggle is enabled and intensity is non-zero
        if enable_prep and abs(prep_intensity) > 1e-4:
            # 1. Resolve preset file and patch model weights
            lora_path = folder_paths.get_full_path("loras", self.PRESET_LORA_NAME)
            if lora_path and os.path.exists(lora_path):
                lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
                patched_model, patched_clip = comfy.sd.load_lora_for_models(
                    model, clip, lora_data, prep_intensity, prep_intensity
                )
            else:
                print(f"[BaseCanvasPrepper] Notice: '{self.PRESET_LORA_NAME}' not found. Passing through unmodified.")

            # 2. Prepend base stage triggers
            triggers = [self.DEFAULT_TRIGGER_WORDS]
            if extra_keywords.strip():
                triggers.append(extra_keywords.strip())
            trigger_str = ", ".join(triggers)

            if final_prompt:
                final_prompt = f"{trigger_str}, {final_prompt}"
            else:
                final_prompt = trigger_str

        # 3. Encode tokens into Conditioning format
        tokens = patched_clip.tokenize(final_prompt)
        cond, pooled = patched_clip.encode_from_tokens(tokens, return_pooled=True)
        conditioning = [[cond, {"pooled_output": pooled}]]

        return (patched_model, patched_clip, conditioning, final_prompt)