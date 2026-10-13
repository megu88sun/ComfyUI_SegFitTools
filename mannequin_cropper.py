# ==============================================================================
# File Name:    mannequin_cropper.py
# Description:  Crops mannequin upper/lower images using DensePose geometry.
# Proposer:     MeguPaPa3
# Author:       OpenAI Codex
# Created:      2026-10-10
#
# Change Log:
# Date        Author        Description
# ----------  ------------  --------------------------------------------------
# 2026-10-10  OpenAI Codex  Initial release.
# 2026-10-10  OpenAI Codex  Added independent upper/lower Y offsets.
# 2026-10-13  OpenAI Codex  Removed the premium Body Line Cropper dependency.
# ==============================================================================
#

from __future__ import annotations

import json
from typing import Any

import torch

try:
    from .densepose_crop_geometry import DensePoseCropGeometry
except ImportError:
    from densepose_crop_geometry import DensePoseCropGeometry


class MannequinCropper:
    """Locate Body Line regions from DensePose and crop the mannequin image."""

    CATEGORY = "SegFitTools/Preprocess"
    FUNCTION = "crop_mannequin"
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("upper_image", "lower_image", "crop_data")

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        return {
            "required": {
                "densepose_image": ("IMAGE",),
                "mannequin_image": ("IMAGE",),
                "upper_bottom_margin": (
                    "FLOAT",
                    {"default": 0.03, "min": 0.0, "max": 0.20, "step": 0.01},
                ),
                "lower_side_margin": (
                    "FLOAT",
                    {"default": 0.10, "min": 0.0, "max": 0.20, "step": 0.01},
                ),
                "upper_y_offset": (
                    "INT", {"default": 0, "min": -5, "max": 5, "step": 1}
                ),
                "lower_y_offset": (
                    "INT", {"default": 0, "min": -200, "max": 0, "step": 5}
                ),
            }
        }

    @staticmethod
    def _validate_inputs(
        densepose_image: torch.Tensor, mannequin_image: torch.Tensor
    ) -> None:
        """Require aligned ComfyUI image batches without implicit resizing."""
        for name, image in (
            ("densepose_image", densepose_image),
            ("mannequin_image", mannequin_image),
        ):
            if not isinstance(image, torch.Tensor):
                raise TypeError(
                    f"[MannequinCropper] {name} must be a torch.Tensor."
                )
            if image.ndim != 4 or image.shape[-1] < 3:
                raise ValueError(f"[MannequinCropper] {name} must be [B,H,W,C].")
            if not torch.isfinite(image).all():
                raise ValueError(
                    f"[MannequinCropper] {name} contains non-finite pixels."
                )
        if densepose_image.shape[:3] != mannequin_image.shape[:3]:
            raise ValueError(
                "[MannequinCropper] DensePose and mannequin image must have "
                "identical batch, height, and width."
            )

    @staticmethod
    def _stack(crops: list[torch.Tensor], name: str) -> torch.Tensor:
        """Form one ComfyUI batch when every crop has the same dimensions."""
        if len({tuple(crop.shape) for crop in crops}) != 1:
            raise ValueError(
                f"[MannequinCropper] Batch produced different {name} sizes."
            )
        return torch.stack(crops)

    @staticmethod
    def _shift_crop_y(
        crop: dict[str, int], offset: int, image_height: int
    ) -> dict[str, int]:
        """Move a crop vertically while preserving its original height."""
        crop_height = crop["y2"] - crop["y1"]
        # The adjustment is expressed relative to the DensePose baseline:
        # negative values move down and positive values move up.
        y1 = max(0, min(crop["y1"] - int(offset), image_height - crop_height))
        shifted = dict(crop)
        shifted["y1"] = y1
        shifted["y2"] = y1 + crop_height
        return shifted

    @classmethod
    def crop_mannequin(
        cls,
        densepose_image: torch.Tensor,
        mannequin_image: torch.Tensor,
        upper_bottom_margin: float,
        lower_side_margin: float,
        upper_y_offset: int = 0,
        lower_y_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Return mannequin pixels cropped by DensePose-derived coordinates."""
        cls._validate_inputs(densepose_image, mannequin_image)
        upper_crops = []
        lower_crops = []
        locations = []

        # Shared geometry gives free and premium croppers identical coordinates.
        for index in range(densepose_image.shape[0]):
            rgb = DensePoseCropGeometry.rgb8(densepose_image[index])
            located = DensePoseCropGeometry.locate(
                rgb, upper_bottom_margin, lower_side_margin
            )
            image_height = located["image"]["height"]
            upper = cls._shift_crop_y(
                located["upper_crop"], upper_y_offset, image_height)
            lower = cls._shift_crop_y(
                located["lower_crop"], lower_y_offset, image_height)
            located["upper_crop"] = upper
            located["lower_crop"] = lower
            upper_crops.append(
                DensePoseCropGeometry.crop_image(mannequin_image, index, upper)
            )
            lower_crops.append(
                DensePoseCropGeometry.crop_image(mannequin_image, index, lower)
            )
            locations.append(located)

        payload = locations[0] if len(locations) == 1 else {"batch": locations}
        crop_data = json.dumps(payload, indent=2, ensure_ascii=False)
        return (
            cls._stack(upper_crops, "upper crop"),
            cls._stack(lower_crops, "lower crop"),
            crop_data,
        )
