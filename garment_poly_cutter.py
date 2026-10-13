# ==============================================================================
# File Name  : garment_poly_cutter.py
# Description: Interactive polygon cutting backend node for extracting garments.
# Proposer   : MeguPaPa3
# Author     : Google Gemini
# Created    : 2026-09-18
#
# Change Log:
# Date       Author         Description
# ---------- -------------- ----------------------------------------------------
# 2026-09-18 Google Gemini  Initial unified implementation matching FitGarmentPro
#                           conventions with strict type guards and error checks.
# 2026-09-18 Google Gemini  Added folder_paths import fallback to support isolated
#                           pytest runs outside full ComfyUI runtime.
# 2026-09-28 Google Gemini  Adapted to AGENTS.md Rev.0.0.5: Added MockFolderPaths
#                           fallback, compressed multi-line arguments, and 
#                           refined tensor operations.
# 2026-10-08 OpenAI Codex  Added detector-option-aware locked raster masks.
# ==============================================================================

from __future__ import annotations
import json
import os
import tempfile
from typing import Any
import uuid
import cv2
import numpy as np
from PIL import Image
import torch

# Use package-relative import in ComfyUI and retain standalone test support.
try:
    from .underwear_mask_restorer import UnderwearMaskRestorer
except ImportError:
    from underwear_mask_restorer import UnderwearMaskRestorer

# The Rules of Import: Safe fallback for standalone test execution
try:
    import folder_paths
except ImportError:
    class MockFolderPaths:
        @staticmethod
        def get_temp_directory() -> str:
            return os.path.join(tempfile.gettempdir(), "comfyui_temp")
    folder_paths = MockFolderPaths()


class GarmentPolyCutter:
    CATEGORY = "SegFitTools/Cutter"
    FUNCTION = "cut_garments"
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "MASK", "IMAGE", "MASK")
    RETURN_NAMES = (
        "person_image",
        "combined_mask",
        "bra_crop",
        "bra_mask",
        "panties_crop",
        "panties_mask",
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "person_image": ("IMAGE",),
                "poly_state": ("STRING", {"default": "", "multiline": True}),
                "action": (["setup", "run"], {"default": "setup"}),
            },
            "optional": {
                "bra_mask": ("MASK",),
                "panties_mask": ("MASK",),
                "bra_option": ("STRING", {"forceInput": True}),
                "panties_option": ("STRING", {"forceInput": True}),
                "crop_data": ("STRING", {"forceInput": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def VALIDATE_INPUTS(
        cls, person_image: Any = None, poly_state: str = "", **_kwargs: Any
    ) -> bool | str:
        if person_image is not None and (not isinstance(person_image, torch.Tensor) or person_image.ndim != 4):
            return "[GarmentPolyCutter] 'person_image' must be an IMAGE tensor (B, H, W, C)."
        if poly_state and str(poly_state).strip():
            try:
                json.loads(poly_state)
            except json.JSONDecodeError as exc:
                return f"[GarmentPolyCutter] 'poly_state' must be valid JSON: {exc.msg}"
        return True

    @staticmethod
    def _default_state() -> dict[str, Any]:
        return {"version": 1, "selected": "bra", "bra_points": [], "panties_points": []}

    @classmethod
    def _parse_state(cls, value: str | None) -> dict[str, Any]:
        default = cls._default_state()
        if not value or not str(value).strip():
            return default
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return default
        loaded.setdefault("bra_points", [])
        loaded.setdefault("panties_points", [])
        loaded.setdefault("selected", "bra")
        return loaded

    @staticmethod
    def _rasterize_polygon(points: list[list[float]], height: int, width: int) -> torch.Tensor:
        # Convert normalized floating point coordinates to binary polygon mask
        mask = np.zeros((height, width), dtype=np.uint8)
        if len(points) >= 3:
            pts = np.array([[round(p[0] * width), round(p[1] * height)] for p in points], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        return torch.from_numpy(mask).float() / 255.0

    @staticmethod
    def _crop_with_mask(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Bounding box isolation based on non-zero mask pixels
        nz = torch.nonzero(mask > 0.5)
        if nz.shape[0] == 0:
            return torch.zeros((1, 64, 64, image.shape[-1])), torch.zeros((1, 64, 64))
        y0, y1 = nz[:, 0].min().item(), nz[:, 0].max().item() + 1
        x0, x1 = nz[:, 1].min().item(), nz[:, 1].max().item() + 1
        return image[y0:y1, x0:x1].unsqueeze(0), mask[y0:y1, x0:x1].unsqueeze(0)

    @staticmethod
    def _save_ui_image(image: torch.Tensor, prefix: str) -> dict[str, str]:
        # Export background image into temporary folder for frontend canvas rendering
        array = (image[0, ..., :3].detach().cpu().clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
        filename = f"{prefix}_{uuid.uuid4().hex}.png"
        directory = folder_paths.get_temp_directory()
        os.makedirs(directory, exist_ok=True)
        Image.fromarray(array, mode="RGB").save(os.path.join(directory, filename), compress_level=1)
        return {"filename": filename, "subfolder": "", "type": "temp"}

    @staticmethod
    def _save_ui_mask(mask: torch.Tensor, prefix: str, color: tuple[int, int, int]) -> dict[str, str]:
        # Keep the established raster mask exact; this image is display-only.
        binary = (mask.detach().cpu().numpy() > 0.5).astype(np.uint8)
        edge = binary - cv2.erode(binary, np.ones((3, 3), dtype=np.uint8))
        rgba = np.zeros((*binary.shape, 4), dtype=np.uint8)
        rgba[..., :3] = np.asarray(color, dtype=np.uint8)
        rgba[..., 3] = binary * 76
        rgba[..., 3] = np.maximum(rgba[..., 3], edge * 230)
        filename = f"{prefix}_{uuid.uuid4().hex}.png"
        directory = folder_paths.get_temp_directory()
        os.makedirs(directory, exist_ok=True)
        Image.fromarray(rgba, mode="RGBA").save(os.path.join(directory, filename), compress_level=1)
        return {"filename": filename, "subfolder": "", "type": "temp"}

    @staticmethod
    def _restore_locked_mask(
        person: torch.Tensor, mask: Any, option: str, crop_data: str,
        crop_name: str, label: str,
    ) -> tuple[torch.Tensor, bool]:
        height, width = person.shape[:2]
        empty = torch.zeros((height, width), device=person.device, dtype=person.dtype)
        if option != "MASK_READY":
            return empty, False
        if not isinstance(mask, torch.Tensor) or mask.ndim != 3 or mask.shape[0] < 1:
            raise ValueError(f"{label}_option is MASK_READY but {label}_mask is invalid.")
        if not bool(torch.count_nonzero(mask[0])):
            raise ValueError(f"{label}_option is MASK_READY but {label}_mask is empty.")

        records = UnderwearMaskRestorer._parse_crop_data(crop_data)
        source_height, source_width = UnderwearMaskRestorer._canvas_size(records[0])
        if (source_height, source_width) != (height, width):
            raise ValueError(f"{label}_mask source canvas does not match person_image.")
        box = UnderwearMaskRestorer._crop_box(records[0], crop_name, height, width)
        expected = (box[3] - box[1], box[2] - box[0])
        if tuple(mask[0].shape) != expected:
            raise ValueError(f"{label}_mask size does not match {crop_name}.")

        # Restore only the established side so the other side remains editable.
        restored = empty.clone()
        source = mask[0].to(device=person.device, dtype=person.dtype).clamp(0.0, 1.0)
        restored[box[1]:box[3], box[0]:box[2]] = source
        return restored, True

    def cut_garments(
        self,
        person_image: torch.Tensor,
        poly_state: str,
        action: str,
        unique_id: str | None = None,
        bra_mask: torch.Tensor | None = None,
        panties_mask: torch.Tensor | None = None,
        bra_option: str = "POLY_CUTTER",
        panties_option: str = "POLY_CUTTER",
        crop_data: str = "",
    ) -> dict[str, Any]:
        try:
            if person_image.ndim != 4 or person_image.shape[0] < 1:
                raise ValueError("person_image must contain at least one image.")

            person = person_image[0]
            height, width = person.shape[:2]
            state = self._parse_state(poly_state)
            # Detector-ready sides remain exact raster masks and are not
            # converted into editable polygon points.
            locked_bra, bra_locked = self._restore_locked_mask(
                person, bra_mask, bra_option, crop_data, "upper_crop", "bra")
            locked_panties, panties_locked = self._restore_locked_mask(
                person, panties_mask, panties_option, crop_data, "lower_crop", "panties")

            # Rasterize bra and panties masks independently
            bra_full_mask = self._rasterize_polygon(
                state["bra_points"], height, width).to(person.device)
            panties_full_mask = self._rasterize_polygon(
                state["panties_points"], height, width).to(person.device)
            if bra_locked:
                bra_full_mask = locked_bra
            if panties_locked:
                panties_full_mask = locked_panties
            combined_mask = torch.maximum(bra_full_mask, panties_full_mask)

            # Extract tight bounding-box crops for both items
            bra_crop, bra_crop_mask = self._crop_with_mask(person, bra_full_mask)
            panties_crop, panties_crop_mask = self._crop_with_mask(
                person, panties_full_mask)

            ui = {
                "node_id": str(unique_id or ""),
                "state": state,
                "person": self._save_ui_image(person.unsqueeze(0), "segfit_poly_person"),
                "locks": {"bra": bra_locked, "panties": panties_locked},
                "bra_locked": self._save_ui_mask(
                    bra_full_mask, "segfit_poly_bra", (84, 199, 236)) if bra_locked else None,
                "panties_locked": self._save_ui_mask(
                    panties_full_mask, "segfit_poly_panties", (243, 156, 221)) if panties_locked else None,
            }

            result = (
                person.unsqueeze(0),
                combined_mask.unsqueeze(0),
                bra_crop,
                bra_crop_mask,
                panties_crop,
                panties_crop_mask,
            )

            # A fully established pair belongs exclusively to Restorer. This
            # fallback emits no data if invoked outside route-aware lazy flow.
            if bra_locked and panties_locked:
                result = (result[0], None, None, None, None, None)

            # Both established masks bypass manual editing. One or two open
            # sides retain setup blocking until the editor RUN button is used.
            if action != "run" and not (bra_locked and panties_locked):
                try:
                    from comfy_execution.graph_utils import ExecutionBlocker
                except ImportError:
                    try:
                        from comfy_execution.graph import ExecutionBlocker
                    except ImportError:
                        ExecutionBlocker = lambda x: None  # noqa: E731
                result = (result[0],) + tuple(ExecutionBlocker(None) for _ in result[1:])

            return {"ui": {"segfit_poly": [ui]}, "result": result}
        except Exception as exc:
            raise RuntimeError(f"[GarmentPolyCutter] Execution failed: {exc}") from None
