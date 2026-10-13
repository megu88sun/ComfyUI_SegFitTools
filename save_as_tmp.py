# ==============================================================================
# File Name:    save_as_tmp.py
# Description:  Saves temporary IMAGE batches as non-image NumPy data.
# Proposer:     MeguPaPa3
# Author:       Google Gemini
# Created:      2026-09-21
#
# Change Log:
# Date        Author         Description
# ----------  -------------  --------------------------------------------------
# 2026-09-21  Google Gemini  Initial release adhering to AGENTS.md guidelines.
# 2026-10-01  OpenAI Codex  Changed storage to source-named, uncompressed NumPy
#                            data for the dedicated LoadTmp node.
# ==============================================================================

import os
import re

import numpy as np
import torch

# The Rules of Import: Using try-except to Pass Standalone Tests
# Mocking folder_paths is required when executed outside ComfyUI environment.
try:
    import folder_paths
except ImportError:
    class MockFolderPaths:
        @staticmethod
        def get_input_directory():
            return os.path.abspath(os.path.join(os.path.dirname(__file__), "mock_input"))

    folder_paths = MockFolderPaths()


INVALID_FILENAME_CHARACTERS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _safe_name(value, fallback):
    """Sanitizes strings to ensure valid and cross-platform file names."""
    value = INVALID_FILENAME_CHARACTERS.sub("_", str(value).strip())
    value = value.rstrip(". ")
    return value or fallback


class SaveAsTmp:
    """
    Saves an IMAGE batch as uncompressed NumPy data without a UI preview.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename_stem": ("STRING", {"default": ""}),
                "output_folder": ("STRING", {"default": "SegFit_TEMP"}),
            }
        }

    CATEGORY = "SegFitTools/Image"
    # Pass-through IMAGE tensor allows chaining directly into cache clearance nodes
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "saved_filename", "saved_path")
    FUNCTION = "save_tmp"
    OUTPUT_NODE = True

    # The fixed suffix keeps SaveTmp and LoadTmp naming synchronized.
    INTERNAL_SUFFIX = "mannequin"
    FILE_EXTENSION = "tmp"
    ON_CONFLICT = "overwrite"

    def save_tmp(self, image, filename_stem, output_folder):
        input_dir = folder_paths.get_input_directory()
        saved_path = self._save_image_batch(image, input_dir, output_folder, filename_stem)

        # Completely suppress {"ui": {"images": ...}} payload to prevent frontend preview
        return (image, os.path.basename(saved_path), saved_path)

    @classmethod
    def _save_image_batch(cls, image, input_dir, output_folder, filename_stem):
        output_dir = cls._resolve_output_directory(input_dir, output_folder)
        os.makedirs(output_dir, exist_ok=True)

        cls._normalize_images(image)
        filename = cls._build_filename(filename_stem)
        full_path = cls._select_path(
            output_dir, os.path.splitext(filename)[0], cls.FILE_EXTENSION, cls.ON_CONFLICT
        )

        # ComfyUI Tensor Convention: [B, H, W, C], Codomain is 0.0-1.0.
        # Store the complete batch as uint8 NPY data without PNG encoding or compression.
        array = (image.detach().float().clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)

        # Write through a temporary sibling so an interrupted save cannot leave a partial cache.
        writing_path = f"{full_path}.writing"
        try:
            with open(writing_path, "wb") as file:
                np.save(file, array, allow_pickle=False)
            os.replace(writing_path, full_path)
        finally:
            if os.path.exists(writing_path):
                os.remove(writing_path)

        return full_path

    @classmethod
    def _build_filename(cls, filename_stem):
        safe_stem = _safe_name(filename_stem, "")
        if not safe_stem:
            raise ValueError("[Save As Tmp] filename_stem must not be empty")
        return f"{safe_stem}__{cls.INTERNAL_SUFFIX}.{cls.FILE_EXTENSION}"

    @staticmethod
    def _normalize_images(image):
        if not isinstance(image, torch.Tensor):
            raise TypeError("image must be a torch.Tensor")
        if image.ndim != 4 or image.shape[-1] not in (1, 3, 4):
            raise ValueError("[Save As Tmp] IMAGE must have shape [B,H,W,1], [B,H,W,3], or [B,H,W,4].")
        return image

    @staticmethod
    def _resolve_output_directory(input_dir, output_folder):
        if os.path.isabs(output_folder):
            raise ValueError("output_folder must be relative to the ComfyUI input directory")

        input_root = os.path.abspath(input_dir)
        output_dir = os.path.abspath(os.path.join(input_root, output_folder))

        # Restrict destination path strictly within input directory tree
        if os.path.commonpath([input_root, output_dir]) != input_root:
            raise ValueError("output_folder must remain inside the ComfyUI input directory")

        return output_dir

    @staticmethod
    def _select_path(output_dir, base_name, ext, on_conflict):
        base_path = os.path.join(output_dir, f"{base_name}.{ext}")

        if on_conflict == "overwrite":
            return base_path
        if on_conflict == "error":
            if os.path.exists(base_path):
                raise FileExistsError(f"File already exists: {base_path}")
            return base_path
        if on_conflict == "increment":
            revision = 1
            while True:
                path = os.path.join(output_dir, f"{base_name}__{revision:03d}.{ext}")
                if not os.path.exists(path):
                    return path
                revision += 1

        raise ValueError(f"Unknown on_conflict mode: {on_conflict}")
