# ==============================================================================
# File Name:    save_mask_named.py
# Description:  Saves masks with source-based names and explicit conflict rules.
# Proposer:     MeguPaPa3
# Author:       OpenAI Codex
# Created:      2026-07-21
#
# Change Log:
# Date        Author        Description
# ----------  ------------  --------------------------------------------------
# 2026-07-21  OpenAI Codex  Initial release.
# ==============================================================================

import os
import re

import numpy as np
import torch
from PIL import Image


INVALID_FILENAME_CHARACTERS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _safe_name(value, fallback):
    value = INVALID_FILENAME_CHARACTERS.sub("_", str(value).strip())
    value = value.rstrip(". ")
    return value or fallback


# Saves each MASK as an 8-bit grayscale PNG under ComfyUI's input directory.
# The source stem remains the primary identifier, while method and optional
# revision suffixes preserve provenance without losing image-mask association.
class SaveMaskNamed:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "source_filename": ("STRING", {"default": "source.png"}),
                "output_folder": ("STRING", {"default": "SegFit_MASK"}),
                "method": (["ESAM3", "SegFormer", "Manual"], {"default": "ESAM3"}),
                "on_conflict": (["overwrite", "increment", "error"], {"default": "overwrite"}),
            }
        }

    CATEGORY = "SegFitTools/Image"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("saved_filename", "saved_path")
    FUNCTION = "save_mask"
    OUTPUT_NODE = True

    def save_mask(self, mask, source_filename, output_folder, method, on_conflict):
        import folder_paths

        input_dir = folder_paths.get_input_directory()
        saved = self._save_masks(
            mask,
            input_dir,
            output_folder,
            source_filename,
            method,
            on_conflict,
        )

        ui_images = []
        for path in saved:
            relative = os.path.relpath(path, input_dir)
            ui_images.append({
                "filename": os.path.basename(relative),
                "subfolder": os.path.dirname(relative).replace(os.sep, "/"),
                "type": "input",
            })

        filenames = "\n".join(os.path.basename(path) for path in saved)
        paths = "\n".join(saved)
        return {"ui": {"images": ui_images}, "result": (filenames, paths)}

    @classmethod
    def _save_masks(
        cls,
        mask,
        input_dir,
        output_folder,
        source_filename,
        method,
        on_conflict,
    ):
        output_dir = cls._resolve_output_directory(input_dir, output_folder)
        os.makedirs(output_dir, exist_ok=True)

        source_stem = _safe_name(os.path.splitext(os.path.basename(source_filename))[0], "source")
        method_name = _safe_name(method, "Mask")
        base_name = f"{source_stem}__{method_name}"

        masks = cls._normalize_masks(mask)
        saved = []
        for batch_index, item in enumerate(masks):
            item_name = base_name
            if len(masks) > 1:
                item_name = f"{item_name}__batch{batch_index + 1:03d}"
            path = cls._select_path(output_dir, item_name, on_conflict)
            array = (item.clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)
            Image.fromarray(array, mode="L").save(path, format="PNG", compress_level=4)
            saved.append(path)
        return saved

    @staticmethod
    def _normalize_masks(mask):
        if not isinstance(mask, torch.Tensor):
            raise TypeError("mask must be a torch.Tensor")
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim != 3:
            raise ValueError("MASK must have shape [H,W] or [B,H,W]")
        return mask

    @staticmethod
    def _resolve_output_directory(input_dir, output_folder):
        if os.path.isabs(output_folder):
            raise ValueError("output_folder must be relative to the ComfyUI input directory")

        input_root = os.path.abspath(input_dir)
        output_dir = os.path.abspath(os.path.join(input_root, output_folder))
        if os.path.commonpath([input_root, output_dir]) != input_root:
            raise ValueError("output_folder must remain inside the ComfyUI input directory")
        return output_dir

    @staticmethod
    def _select_path(output_dir, base_name, on_conflict):
        base_path = os.path.join(output_dir, f"{base_name}.png")

        if on_conflict == "overwrite":
            return base_path
        if on_conflict == "error":
            if os.path.exists(base_path):
                raise FileExistsError(f"Mask already exists: {base_path}")
            return base_path
        if on_conflict == "increment":
            revision = 1
            while True:
                path = os.path.join(output_dir, f"{base_name}__{revision:03d}.png")
                if not os.path.exists(path):
                    return path
                revision += 1

        raise ValueError(f"Unknown on_conflict mode: {on_conflict}")
