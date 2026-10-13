# ==============================================================================
# File Name:    load_tmp.py
# Description:  Loads source-named NumPy IMAGE caches without a UI preview.
# Proposer:     MeguPaPa3
# Author:       OpenAI Codex
# Created:      2026-10-01
#
# Change Log:
# Date        Author        Description
# ----------  ------------  --------------------------------------------------
# 2026-10-01  OpenAI Codex  Initial release.
# ==============================================================================

import os

import numpy as np
import torch

# The Rules of Import: Using try-except to Pass Standalone Tests
try:
    import folder_paths
except ImportError:
    class MockFolderPaths:
        @staticmethod
        def get_input_directory():
            return os.path.abspath(os.path.join(os.path.dirname(__file__), "mock_input"))

    folder_paths = MockFolderPaths()

try:
    from .save_as_tmp import SaveAsTmp
except ImportError:
    from save_as_tmp import SaveAsTmp


class LoadTmp:
    """Loads a SaveAsTmp NumPy cache without creating a ComfyUI preview."""

    MAX_DIMENSION = 32768
    MAX_ARRAY_BYTES = 512 * 1024 * 1024

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename_stem": ("STRING", {"default": ""}),
                "input_folder": ("STRING", {"default": "SegFit_TEMP"}),
            }
        }

    CATEGORY = "SegFitTools/Image"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "loaded_filename", "loaded_path")
    FUNCTION = "load_tmp"

    def load_tmp(self, filename_stem, input_folder):
        input_dir = folder_paths.get_input_directory()
        path = self._resolve_path(input_dir, input_folder, filename_stem)
        self._require_file(path)

        # Memory-map first so shape and dtype can be rejected before allocating a tensor copy.
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        self._validate_array(array)
        output_image = torch.from_numpy(np.asarray(array).copy()).float().div_(255.0)

        # Match ComfyUI's preferred intermediate device and dtype when available.
        try:
            import comfy.model_management

            dtype = comfy.model_management.intermediate_dtype()
            device = comfy.model_management.intermediate_device()
            output_image = output_image.to(device=device, dtype=dtype)
        except (ImportError, AttributeError):
            pass

        # A plain tuple deliberately omits the {"ui": {"images": ...}} preview payload.
        return (output_image, os.path.basename(path), path)

    @classmethod
    def _resolve_path(cls, input_dir, input_folder, filename_stem):
        directory = SaveAsTmp._resolve_output_directory(input_dir, input_folder)
        return os.path.join(directory, SaveAsTmp._build_filename(filename_stem))

    @classmethod
    def _validate_array(cls, array):
        if array.dtype != np.uint8:
            raise ValueError(f"[Load Tmp] Expected uint8 data, received {array.dtype}")
        if array.ndim != 4 or array.shape[-1] not in (1, 3, 4):
            raise ValueError("[Load Tmp] Data must have shape [B,H,W,1], [B,H,W,3], or [B,H,W,4]")
        if any(size <= 0 for size in array.shape):
            raise ValueError("[Load Tmp] Data dimensions must be greater than zero")
        if array.shape[1] > cls.MAX_DIMENSION or array.shape[2] > cls.MAX_DIMENSION:
            raise ValueError(f"[Load Tmp] Image dimensions exceed {cls.MAX_DIMENSION} pixels")
        if array.nbytes > cls.MAX_ARRAY_BYTES:
            raise ValueError(f"[Load Tmp] Array exceeds {cls.MAX_ARRAY_BYTES} bytes")

    @staticmethod
    def _require_file(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"[Load Tmp] Not Found: {path}")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[Load Tmp] Not a file: {path}")
        return path

    @classmethod
    def IS_CHANGED(cls, filename_stem, input_folder):
        path = cls._resolve_path(
            folder_paths.get_input_directory(), input_folder, filename_stem
        )
        if not os.path.isfile(path):
            return float("nan")
        stat = os.stat(path)
        return f"{path}:{stat.st_mtime_ns}:{stat.st_size}"
