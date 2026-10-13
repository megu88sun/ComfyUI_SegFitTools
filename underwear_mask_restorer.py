# ==============================================================================
# File Name:    underwear_mask_restorer.py
# Description:  Restores cropped bra and panties masks to the source canvas.
# Proposer:     MeguPaPa3
# Author:       OpenAI Codex
# Created:      2026-10-02
#
# Change Log:
# Date        Author        Description
# ----------  ------------  --------------------------------------------------
# 2026-10-02  OpenAI Codex  Initial release.
# 2026-10-08  OpenAI Codex  Require detector options and gate automatic
#                           restoration unless both masks are established.
# 2026-10-08  OpenAI Codex  Return no mask value instead of blocking when the
#                           automatic pair is incomplete.
# 2026-10-08  OpenAI Codex  Output the authoritative Restorer/Poly Cutter route.
# ==============================================================================

from __future__ import annotations

import json
from typing import Any

import torch


class UnderwearMaskRestorer:
    CATEGORY = "SegFitTools/Mask"
    FUNCTION = "restore_masks"
    RETURN_TYPES = ("MASK", "MASK", "MASK", "STRING")
    RETURN_NAMES = (
        "bra_full_mask", "panties_full_mask", "combined_removal_mask", "route")

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        return {
            "required": {
                "bra_mask": ("MASK",),
                "panties_mask": ("MASK",),
                "bra_option": ("STRING", {"forceInput": True}),
                "panties_option": ("STRING", {"forceInput": True}),
                "crop_data": ("STRING", {"forceInput": True}),
            }
        }

    @staticmethod
    def _parse_crop_data(crop_data: str) -> list[dict[str, Any]]:
        try:
            payload = json.loads(crop_data)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "[UnderwearMaskRestorer] crop_data is not valid JSON."
            ) from exc

        records = payload.get("batch") if isinstance(payload, dict) else None
        if records is None:
            records = [payload]
        if not records or not all(isinstance(value, dict) for value in records):
            raise ValueError(
                "[UnderwearMaskRestorer] crop_data contains no crop records."
            )
        return records

    @staticmethod
    def _canvas_size(record: dict[str, Any]) -> tuple[int, int]:
        try:
            width = int(record["image"]["width"])
            height = int(record["image"]["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "[UnderwearMaskRestorer] crop_data has no valid image size."
            ) from exc
        if width <= 0 or height <= 0:
            raise ValueError(
                "[UnderwearMaskRestorer] image size must be positive."
            )
        return height, width

    @staticmethod
    def _crop_box(
        record: dict[str, Any], name: str, height: int, width: int
    ) -> tuple[int, int, int, int]:
        try:
            crop = record[name]
            x1 = int(crop["x1"])
            y1 = int(crop["y1"])
            x2 = int(crop["x2"])
            y2 = int(crop["y2"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"[UnderwearMaskRestorer] crop_data has no valid {name}."
            ) from exc
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(
                f"[UnderwearMaskRestorer] {name} lies outside the source canvas."
            )
        return x1, y1, x2, y2

    @classmethod
    def restore_masks(
        cls, bra_mask: torch.Tensor, panties_mask: torch.Tensor,
        bra_option: str, panties_option: str, crop_data: str,
    ) -> tuple[Any, Any, Any, str]:
        if bra_mask.ndim != 3 or panties_mask.ndim != 3:
            raise ValueError(
                "[UnderwearMaskRestorer] masks must use ComfyUI [B,H,W] format."
            )
        if bra_mask.shape[0] != panties_mask.shape[0]:
            raise ValueError(
                "[UnderwearMaskRestorer] bra and panties batch sizes must match."
            )

        # An incomplete automatic pair emits no value. Mask Auto Switching
        # therefore sees only the Poly Cutter branch, without a black placeholder.
        ready = bra_option == "MASK_READY" and panties_option == "MASK_READY"
        nonempty = (
            bool(torch.all(torch.count_nonzero(bra_mask, dim=(1, 2)) > 0)) and
            bool(torch.all(torch.count_nonzero(panties_mask, dim=(1, 2)) > 0))
        )
        if not ready or not nonempty:
            return None, None, None, "POLY_CUTTER"

        records = cls._parse_crop_data(crop_data)
        if len(records) != bra_mask.shape[0]:
            raise ValueError(
                "[UnderwearMaskRestorer] crop_data batch size does not match masks."
            )

        bra_full = []
        panties_full = []
        canvas_shapes = set()
        for index, record in enumerate(records):
            height, width = cls._canvas_size(record)
            canvas_shapes.add((height, width))
            upper = cls._crop_box(record, "upper_crop", height, width)
            lower = cls._crop_box(record, "lower_crop", height, width)
            upper_size = (upper[3] - upper[1], upper[2] - upper[0])
            lower_size = (lower[3] - lower[1], lower[2] - lower[0])
            if tuple(bra_mask[index].shape) != upper_size:
                raise ValueError(
                    "[UnderwearMaskRestorer] bra_mask size does not match upper_crop."
                )
            if tuple(panties_mask[index].shape) != lower_size:
                raise ValueError(
                    "[UnderwearMaskRestorer] panties_mask size does not match lower_crop."
                )

            # Restore each mask independently so callers can inspect or replace
            # either garment before creating the combined removal mask.
            bra_canvas = torch.zeros(
                (height, width), device=bra_mask.device, dtype=bra_mask.dtype
            )
            panties_canvas = torch.zeros_like(bra_canvas)
            bra_canvas[upper[1]:upper[3], upper[0]:upper[2]] = bra_mask[index]
            restored_panties = panties_mask[index].to(device=bra_mask.device, dtype=bra_mask.dtype)
            panties_canvas[lower[1]:lower[3], lower[0]:lower[2]] = restored_panties
            bra_full.append(bra_canvas)
            panties_full.append(panties_canvas)

        if len(canvas_shapes) != 1:
            raise ValueError(
                "[UnderwearMaskRestorer] Batch members use different canvas sizes."
            )

        bra_result = torch.stack(bra_full).clamp(0.0, 1.0)
        panties_result = torch.stack(panties_full).clamp(0.0, 1.0)
        combined = torch.maximum(bra_result, panties_result)
        return bra_result, panties_result, combined, "RESTORER"
