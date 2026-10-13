# ==============================================================================
# File Name:    debug_body_crop_save.py
# Description:  Crop loaded photos using Cropper JSON and save debug PNGs.
# Proposer:     MeguPaPa3
# Author:       OpenAI Codex
# Created:      2026-10-06
#
# Change Log:
# Date        Author        Description
# ----------  ------------  --------------------------------------------------
# 2026-10-06  OpenAI Codex  Initial temporary JSON crop/save debug node.
# ==============================================================================
#

import json
import os

import torch

try:
    import folder_paths
except ImportError:
    from unittest.mock import MagicMock
    folder_paths = MagicMock()

try:
    from .save_temp_image_named import SaveTempImageNamed, _safe_name
except ImportError:
    from save_temp_image_named import SaveTempImageNamed, _safe_name


class DebugBodyCropSave:
    CATEGORY = "SegFitTools/TEST"
    FUNCTION = "crop_and_save"
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("upper_image", "lower_image", "upper_path", "lower_path")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "crop_data": ("STRING", {"forceInput": True}),
            "filename_stem": ("STRING", {"forceInput": True}),
        }}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Re-save debug files on each queued run even if inputs are unchanged.
        return float("nan")

    @staticmethod
    def _box(record, name, height, width):
        try:
            box = record[name]
            values = tuple(box[key] for key in ("x1", "y1", "x2", "y2"))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"[DebugBodyCropSave] Missing {name} coordinates.") from exc
        if any(type(value) is not int for value in values):
            raise ValueError(f"[DebugBodyCropSave] {name} coordinates must be integers.")
        x1, y1, x2, y2 = values
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"[DebugBodyCropSave] {name} lies outside the image.")
        return values

    @classmethod
    def _crop_images(cls, image, crop_data):
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError("[DebugBodyCropSave] IMAGE must use [B,H,W,C] format.")
        if min(image.shape) <= 0 or image.shape[-1] not in (1, 3, 4):
            raise ValueError("[DebugBodyCropSave] IMAGE must be nonempty with 1, 3 or 4 channels.")
        if not torch.isfinite(image).all():
            raise ValueError("[DebugBodyCropSave] IMAGE contains non-finite pixels.")
        try:
            payload = json.loads(crop_data)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("[DebugBodyCropSave] crop_data is not valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError("[DebugBodyCropSave] crop_data must be a JSON object.")
        records = payload.get("batch", [payload])
        if not isinstance(records, list) or len(records) != image.shape[0]:
            raise ValueError("[DebugBodyCropSave] JSON batch size must match IMAGE.")

        # Use the exact exclusive rectangles from Body Line Cropper.
        # Validate both regions for every image before creating any files.
        height, width = image.shape[1:3]
        crops = {"upper": [], "lower": []}
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError("[DebugBodyCropSave] Invalid crop record.")
            size = record.get("image")
            if not isinstance(size, dict) or size.get("height") != height or size.get("width") != width:
                raise ValueError("[DebugBodyCropSave] Loaded image size differs from the JSON source canvas; no resizing is allowed.")
            for region in crops:
                x1, y1, x2, y2 = cls._box(record, f"{region}_crop", height, width)
                crops[region].append(image[index, y1:y2, x1:x2])
        for region, items in crops.items():
            if len({tuple(item.shape) for item in items}) != 1:
                raise ValueError(f"[DebugBodyCropSave] {region} batch crop sizes differ.")
        return torch.stack(crops["upper"]), torch.stack(crops["lower"])

    def crop_and_save(self, image, crop_data, filename_stem):
        upper, lower = self._crop_images(image, crop_data)
        input_dir = folder_paths.get_input_directory()
        # filename_stem is already extension-free. Add a temporary extension
        # for the shared saver so a dotted stem is not shortened a second time.
        source_name = _safe_name(filename_stem, "source") + ".png"
        upper_paths = SaveTempImageNamed._save_images(upper, input_dir, "SegFit_TEST", source_name, "upper")
        lower_paths = SaveTempImageNamed._save_images(lower, input_dir, "SegFit_TEST", source_name, "lower")
        previews = [{"filename": os.path.basename(path), "subfolder": "SegFit_TEST", "type": "input"}
                    for path in upper_paths + lower_paths]
        return {"ui": {"images": previews}, "result": (upper, lower, "\n".join(upper_paths), "\n".join(lower_paths))}
