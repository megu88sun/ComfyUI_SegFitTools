# ==============================================================================
# File Name:    densepose_cropper.py
# Description:  Creates Body Line crop coordinates from DensePose only.
# Proposer:     MeguPaPa3
# Author:       OpenAI Codex
# Created:      2026-10-13
#
# Change Log:
# Date        Author        Description
# ----------  ------------  --------------------------------------------------
# 2026-10-13  OpenAI Codex  Initial release.
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


class DensePoseCropper:
    """Return Body Line crop geometry without requiring evidence images."""

    CATEGORY = "SegFitTools/Preprocess"
    FUNCTION = "create_crop_data"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("crop_data",)

    _UPPER_BOTTOM_MARGIN = 0.03
    _LOWER_SIDE_MARGIN = 0.10

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        return {"required": {"densepose": ("IMAGE",)}}

    @staticmethod
    def _validate_densepose(densepose: torch.Tensor) -> None:
        """Validate the ComfyUI IMAGE tensor used for geometry detection."""
        if not isinstance(densepose, torch.Tensor):
            raise TypeError("[DensePoseCropper] densepose must be a torch.Tensor.")
        if densepose.ndim != 4 or densepose.shape[-1] < 3:
            raise ValueError("[DensePoseCropper] densepose must be [B,H,W,C].")
        if densepose.shape[0] == 0:
            raise ValueError("[DensePoseCropper] densepose batch must not be empty.")
        if not torch.isfinite(densepose).all():
            raise ValueError("[DensePoseCropper] densepose contains non-finite pixels.")

    @classmethod
    def create_crop_data(cls, densepose: torch.Tensor) -> tuple[str]:
        """Create the standard upper/lower crop JSON for each batch image."""
        cls._validate_densepose(densepose)
        locations = []

        # Shared geometry keeps free and premium workflows interchangeable.
        for index in range(densepose.shape[0]):
            rgb = DensePoseCropGeometry.rgb8(densepose[index])
            locations.append(
                DensePoseCropGeometry.locate(
                    rgb, cls._UPPER_BOTTOM_MARGIN, cls._LOWER_SIDE_MARGIN
                )
            )

        payload = locations[0] if len(locations) == 1 else {"batch": locations}
        return (json.dumps(payload, indent=2, ensure_ascii=False),)
