# ==============================================================================
# File Name:    load_image_by_chain.py
# Description:  Loads an image from chained folder, stem, and suffix strings.
# Proposer:     MeguPaPa3
# Author:       OpenAI Codex
# Created:      2026-07-24
#
# Change Log:
# Date        Author        Description
# ----------  ------------  --------------------------------------------------
# 2026-07-24  OpenAI Codex  Initial release.
# 2026-07-24  OpenAI Codex  Changed extension input to a verbatim suffix and extension.
# 2026-07-24  OpenAI Codex  Added explicit workflow-stopping Not Found handling.
# 2026-07-24  OpenAI Codex  Moved file validation to runtime for linked STRING inputs.
# 2026-07-26  OpenAI Codex  Added optional in-node preview generation.
# ==============================================================================

import os
import uuid

import numpy as np
from PIL import Image

from .load_image_no_upload import LoadImageNoUpload


# Resolves a path from chainable STRING values, preserves RGB values underneath
# transparent pixels, returns the inverted alpha MASK, and publishes a temp PNG
# so ComfyUI can display the loaded image inside the node.
class LoadImageByChain:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_name": ("STRING", {"default": "SegFit_MASK"}),
                "filename": ("STRING", {"default": ""}),
                "suffix_and_extension": ("STRING", {"default": ".png"}),
                "preview": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "on",
                        "label_off": "off",
                    },
                ),
            }
        }

    CATEGORY = "SegFitTools/Image"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "load_image"

    # def load_image(self, folder_name, filename, suffix_and_extension):
    def load_image(self, folder_name, filename, suffix_and_extension, preview=True):
        import folder_paths

        image_path = self._resolve_path(
            folder_paths.get_input_directory(), folder_name, filename, suffix_and_extension
        )
        self._require_file(image_path)

        output_image, output_mask = LoadImageNoUpload._decode_image(image_path)

        try:
            import comfy.model_management

            dtype = comfy.model_management.intermediate_dtype()
            device = comfy.model_management.intermediate_device()
            output_image = output_image.to(device=device, dtype=dtype)
            output_mask = output_mask.to(device=device, dtype=dtype)
        except (ImportError, AttributeError):
            pass

        if preview:
            preview_path = self._save_preview(
                output_image[0],
                output_mask[0],
                folder_paths.get_temp_directory(),
            )
            return {
                "ui": {
                    "images": [{
                        "filename": os.path.basename(preview_path),
                        "subfolder": "",
                        "type": "temp",
                    }]
                },
                "result": (output_image, output_mask),
            }

        return {"result": (output_image, output_mask)}

    @staticmethod
    def _resolve_path(input_dir, folder_name, filename, suffix_and_extension):
        folder_name = str(folder_name or "").strip().strip('"')
        filename = str(filename or "").strip().strip('"')
        suffix_and_extension = str(suffix_and_extension or "").strip()

        if not filename:
            candidate = folder_name
        else:
            _, current_extension = os.path.splitext(filename)
            if not current_extension:
                filename = f"{filename}{suffix_and_extension}"
            candidate = os.path.join(folder_name, filename)

        if not candidate:
            raise ValueError("folder_name or filename must identify an image file")
        if not os.path.isabs(candidate):
            candidate = os.path.join(input_dir, candidate)
        return os.path.abspath(os.path.normpath(candidate))

    @staticmethod
    def _save_preview(image, mask, temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
        rgb = (image[..., :3].detach().float().clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)
        alpha = ((1.0 - mask.detach().float().clamp(0.0, 1.0)).cpu().numpy() * 255.0).round().astype(np.uint8)
        rgba = np.concatenate((rgb, alpha[..., None]), axis=-1)
        path = os.path.join(temp_dir, f"segfit_chain_{uuid.uuid4().hex}.png")
        Image.fromarray(rgba, mode="RGBA").save(path, format="PNG", compress_level=1)
        return path

    @staticmethod
    def _require_file(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"[Load Image By Chain] Not Found: {path}")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[Load Image By Chain] Not a file: {path}")
        return path

    @classmethod
    # def IS_CHANGED(cls, folder_name, filename, suffix_and_extension):
    def IS_CHANGED(cls, folder_name, filename, suffix_and_extension, preview=True):
        import folder_paths

        path = cls._resolve_path(
            folder_paths.get_input_directory(), folder_name, filename, suffix_and_extension
        )
        if not os.path.isfile(path):
            return float("nan")
        stat = os.stat(path)
        return f"{path}:{stat.st_mtime_ns}:{stat.st_size}"
