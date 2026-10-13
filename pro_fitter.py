# ==============================================================================
# File Name:    pro_fitter.py
# Description:  Provides the interactive Fit Garment Pro backend node with
#               robust tensor normalization, affine placement, and UI caching.
# Proposer:     MeguPaPa3
# Author:       Google Gemini
# Created:      2026-07-16
#
# Change Log:
# Date        Author         Description
# ----------  ------------  --------------------------------------------------
# 2026-07-16  OpenAI Codex  Initial release.
# 2026-07-17  OpenAI Codex  Updated editor state, mask outputs, transforms.
# 2026-09-18  Google Gemini Refactor: add VALIDATE_INPUTS, type annotations,
#                           top-level imports, and graceful out-of-bounds guards.
# ==============================================================================
"""Interactive, engine-agnostic garment placement backend."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import numpy as np
import torch
from PIL import Image

# import folder_paths
# 2026.09.18 update
try:
    import folder_paths
except ImportError:
    folder_paths = None
from .advanced_fitter import FitGarmentAdvanced


class FitGarmentPro(FitGarmentAdvanced):
    CATEGORY = "SegFitTools/Fit"
    FUNCTION = "fit_pro"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "MASK", "MASK")
    RETURN_NAMES = (
        "original_person_image",
        "fitting_guidance",
        "placement_mask",
        "original_underwear_mask",
        "replace_mask",
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "person_image": ("IMAGE",),
                "bra_image": ("IMAGE",),
                "panties_image": ("IMAGE",),
                "transform_state": ("STRING", {"default": "", "multiline": True}),
                "action": (["setup", "run"], {"default": "setup"}),
            },
            "optional": {
                "bra_mask": ("MASK",),
                "panties_mask": ("MASK",),
                "original_underwear_mask": ("MASK",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        person_image: Any = None,
        bra_image: Any = None,
        panties_image: Any = None,
        transform_state: str = "",
        **_kwargs: Any,
    ) -> bool | str:
        for name, tensor in (
            ("person_image", person_image),
            ("bra_image", bra_image),
            ("panties_image", panties_image),
        ):
            if tensor is not None and (
                not isinstance(tensor, torch.Tensor) or tensor.ndim != 4
            ):
                return f"[FitGarmentPro] {name} must be a 4D IMAGE tensor (B,H,W,C)."

        if transform_state and str(transform_state).strip():
            try:
                json.loads(transform_state)
            except json.JSONDecodeError as exc:
                return f"[FitGarmentPro] transform_state must be valid JSON: {exc.msg}"

        return True

    @staticmethod
    def _infer_mask(image: torch.Tensor) -> torch.Tensor:
        rgb = image[..., :3]
        edge = torch.cat((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), dim=0)
        background = edge.median(dim=0).values
        distance = torch.sqrt(torch.mean((rgb - background) ** 2, dim=-1))
        mask = (distance > 0.035).float()
        if mask.sum() < 16:
            mask = (rgb.abs().amax(dim=-1) > 0.01).float()
        return mask

    def _prepare_source(
        self, image: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 4:
            raise ValueError("[FitGarmentPro] Images must have shape (B,H,W,C).")
        source = image[0]
        if mask is None:
            source_mask = self._infer_mask(source)
        else:
            source_mask = self._as_mask(mask)[0]
            source_h, source_w = source.shape[:2]
            if tuple(source_mask.shape) != (source_h, source_w):
                source_mask = self._resize_mask(
                    source_mask[None], source_h, source_w
                )[0]
            # ComfyUI Load Image exposes PNG alpha as inverted:
            # Detect predominantly white edge and normalize.
            edge = torch.cat((
                source_mask[0], source_mask[-1],
                source_mask[:, 0], source_mask[:, -1],
            ))
            if edge.float().mean() > 0.5:
                source_mask = 1.0 - source_mask

        x0, y0, x1, y1 = self._bbox(source_mask, 0.5)
        crop = source[y0:y1, x0:x1]
        crop_mask = source_mask[y0:y1, x0:x1]
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            raise ValueError(
                "[FitGarmentPro] Garment mask does not overlap the garment image after cropping."
            )
        return crop, crop_mask

    @staticmethod
    def _default_state() -> dict[str, Any]:
        return {
            "version": 1,
            "selected": "bra",
            "bra": {"cx": 0.50, "cy": 0.38, "width": 0.46, "scale_y": 1.0, "rotation": 0.0},
            "panties": {"cx": 0.50, "cy": 0.70, "width": 0.38, "scale_y": 1.0, "rotation": 0.0},
        }

    @classmethod
    def _parse_state(cls, value: str | None) -> dict[str, Any]:
        default = cls._default_state()
        if not value or not str(value).strip():
            return default
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"[FitGarmentPro] transform_state is not valid JSON: {exc.msg}") from None

        for garment in ("bra", "panties"):
            current = loaded.setdefault(garment, {})
            for key, fallback in default[garment].items():
                current.setdefault(key, fallback)
        loaded.setdefault("version", 1)
        loaded.setdefault("selected", "bra")
        return loaded

    def _place(
        self,
        canvas: torch.Tensor,
        crop: torch.Tensor,
        crop_mask: torch.Tensor,
        settings: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        canvas_h, canvas_w = canvas.shape[:2]
        source_h, source_w = crop.shape[:2]
        if source_h < 1 or source_w < 1:
            raise ValueError("[FitGarmentPro] The prepared garment crop is empty.")

        width = max(1, round(float(settings["width"]) * canvas_w))
        height = max(1, round(width * source_h / source_w * float(settings["scale_y"])))
        fitted = self._resize_image(crop[None], height, width)[0]
        alpha = self._resize_mask(crop_mask[None], height, width)[0]

        # Invert angle to match canvas Y-down clockwise coordinates
        fitted, alpha = self._rotate_expanded(fitted, alpha, -float(settings["rotation"]))
        height, width = alpha.shape
        x = round(float(settings["cx"]) * canvas_w - width / 2)
        y = round(float(settings["cy"]) * canvas_h - height / 2)

        dx0, dy0 = max(0, x), max(0, y)
        dx1, dy1 = min(canvas_w, x + width), min(canvas_h, y + height)

        full_layer = torch.zeros_like(canvas)
        full_mask = torch.zeros((canvas_h, canvas_w), device=canvas.device, dtype=canvas.dtype)
        output = canvas.clone()

        # Canvas外に完全に押し出された場合は空要素として安全にスルー
        if dx1 <= dx0 or dy1 <= dy0:
            return output, full_layer, full_mask, fitted

        sx0, sy0 = dx0 - x, dy0 - y
        sx1, sy1 = sx0 + dx1 - dx0, sy0 + dy1 - dy0
        patch = fitted[sy0:sy1, sx0:sx1].to(canvas.device, canvas.dtype)
        patch_mask = alpha[sy0:sy1, sx0:sx1].to(canvas.device, canvas.dtype)

        channels = min(canvas.shape[-1], patch.shape[-1])
        full_layer[dy0:dy1, dx0:dx1, :channels] = patch[..., :channels]
        full_mask[dy0:dy1, dx0:dx1] = patch_mask

        a = patch_mask[..., None]
        output[dy0:dy1, dx0:dx1, :channels] = (
            patch[..., :channels] * a + output[dy0:dy1, dx0:dx1, :channels] * (1.0 - a)
        )
        return output, full_layer, full_mask, fitted

    def _run(
        self,
        person_image: torch.Tensor,
        bra_image: torch.Tensor,
        panties_image: torch.Tensor,
        transform_state: str,
        bra_mask: torch.Tensor | None = None,
        panties_mask: torch.Tensor | None = None,
        original_underwear_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if person_image.ndim != 4 or person_image.shape[0] < 1:
            raise ValueError("[FitGarmentPro] person_image must contain at least one image.")

        state = self._parse_state(transform_state)
        person = person_image[0]
        bra_crop_src, bra_crop_mask = self._prepare_source(bra_image, bra_mask)
        panties_crop_src, panties_crop_mask = self._prepare_source(panties_image, panties_mask)

        after_bra, bra_layer, placed_bra, bra_crop = self._place(
            person, bra_crop_src, bra_crop_mask, state["bra"]
        )
        composite, panties_layer, placed_panties, panties_crop = self._place(
            after_bra, panties_crop_src, panties_crop_mask, state["panties"]
        )
        combined = torch.maximum(placed_bra, placed_panties)

        if original_underwear_mask is None:
            original_mask = torch.zeros_like(combined)
        else:
            original_mask = self._as_mask(original_underwear_mask)[0]
            if tuple(original_mask.shape) != tuple(combined.shape):
                original_mask = self._resize_mask(
                    original_mask[None], combined.shape[0], combined.shape[1]
                )[0]
            original_mask = original_mask.to(combined.device, combined.dtype).clamp(0, 1)

        replace_mask = torch.maximum(original_mask, combined)
        garment_layer = bra_layer * (1.0 - placed_panties[..., None]) + panties_layer

        return {
            "state": state,
            "editor_preview": composite[None],
            "placement_guide": composite[None],
            "person_image": person[None],
            "garment_layer": garment_layer[None],
            "combined_mask": combined[None],
            "bra_mask": placed_bra[None],
            "panties_mask": placed_panties[None],
            "original_underwear_mask": original_mask[None],
            "replace_mask": replace_mask[None],
            "bra_crop": bra_crop[None],
            "panties_crop": panties_crop[None],
            "source_bra": bra_crop_src[None],
            "source_panties": panties_crop_src[None],
            "source_bra_mask": bra_crop_mask[None],
            "source_panties_mask": panties_crop_mask[None],
        }

    @staticmethod
    def _save_ui_image(
        image: torch.Tensor, prefix: str, mask: torch.Tensor | None = None
    ) -> dict[str, str]:
        array = (image[0, ..., :3].detach().cpu().clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
        mode = "RGB"
        if mask is not None:
            alpha = (mask[0].detach().cpu().clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
            array = np.dstack((array, alpha))
            mode = "RGBA"

        filename = f"{prefix}_{uuid.uuid4().hex}.png"
        directory = folder_paths.get_temp_directory()
        os.makedirs(directory, exist_ok=True)
        Image.fromarray(array, mode=mode).save(
            os.path.join(directory, filename), compress_level=1
        )
        return {"filename": filename, "subfolder": "", "type": "temp"}

    def fit_pro(
        self,
        person_image: torch.Tensor,
        bra_image: torch.Tensor,
        panties_image: torch.Tensor,
        transform_state: str,
        action: str,
        bra_mask: torch.Tensor | None = None,
        panties_mask: torch.Tensor | None = None,
        original_underwear_mask: torch.Tensor | None = None,
        unique_id: str | None = None,
    ) -> dict[str, Any]:
        try:
            data = self._run(
                person_image,
                bra_image,
                panties_image,
                transform_state,
                bra_mask=bra_mask,
                panties_mask=panties_mask,
                original_underwear_mask=original_underwear_mask,
            )
            ui = {
                "node_id": str(unique_id or ""),
                "state": data["state"],
                "person": self._save_ui_image(data["person_image"], "segfit_person"),
                "bra": self._save_ui_image(
                    data["source_bra"], "segfit_bra", data["source_bra_mask"]
                ),
                "panties": self._save_ui_image(
                    data["source_panties"], "segfit_panties", data["source_panties_mask"]
                ),
            }
            result = (
                data["person_image"],
                data["placement_guide"],
                data["combined_mask"],
                data["original_underwear_mask"],
                data["replace_mask"],
            )
            if action != "run":
                try:
                    from comfy_execution.graph_utils import ExecutionBlocker
                except ImportError:
                    from comfy_execution.graph import ExecutionBlocker
                result = (result[0],) + tuple(ExecutionBlocker(None) for _ in result[1:])

            return {"ui": {"segfit_pro": [ui]}, "result": result}
        except Exception as exc:
            raise RuntimeError(f"[Fit Garment Pro] Processing failed: {exc}") from None