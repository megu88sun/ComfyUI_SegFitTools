# ==============================================================================
# File Name:    save_temp_image_named.py
# Description:  Saves temporary images with source-based names and overwrites.
# Proposer:     MeguPaPa3
# Author:       MeguPaPa3
# Created:      2026-07-21
#
# Change Log:
# Date        Author        Description
# ----------  ------------  --------------------------------------------------
# 2026-07-25  MeguPaPa3     Initial release.
# 2026-07-26  OpenAI Codex  Fixed output paths, image modes, and batch handling.
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

# Saves IMAGE tensors as temporary PNG files under ComfyUI's input directory.
# Files use stable source-based names and are overwritten on repeated runs.
class SaveTempImageNamed:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "source_filename": ("STRING", {"default": "source.png"}),
                "method": ("STRING", {"default": "TEMP"}),
                "output_folder": ("STRING", {"default": "SegFit_TEMP"}),
            }
        }

    CATEGORY = "SegFitTools/Image"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("saved_filename", "saved_path")
    FUNCTION = "save_image"
    OUTPUT_NODE = True

    def save_image(self, image, source_filename, method, output_folder):
        import folder_paths

        input_dir = folder_paths.get_input_directory()
        saved = self._save_images(
            image,
            input_dir,
            output_folder,
            source_filename,
            method,
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
    def _save_images(cls, image, input_dir, output_folder, source_filename, method):
        output_dir = cls._resolve_output_directory(input_dir, output_folder)
        os.makedirs(output_dir, exist_ok=True)

        # source_stem = _safe_name(os.path.splitext(os.path.basename(source_filename))[0],"source")
        source_stem = _safe_name(
            os.path.splitext(os.path.basename(source_filename))[0],
            "source",
        )
        method_name = _safe_name(method, "Image")
        images = cls._normalize_images(image)
        # if method:
        #     base_name = f"{source_stem}_{method_name}.png"
        # else:
        #     base_name = f"{source_stem}.png"
        base_name = source_stem
        if str(method).strip():
            base_name = f"{base_name}__{method_name}"

        saved = []
        for batch_index, single_image in enumerate(images):
            item_name = base_name
            if len(images) > 1:
                item_name = f"{item_name}__batch{batch_index + 1:03d}"
            full_path = os.path.join(output_dir, f"{item_name}.png")

            # array = (single_image.clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)
            array = (
                single_image.detach().float().clamp(0.0, 1.0).cpu().numpy()
                * 255.0
            ).round().astype(np.uint8)
            if array.ndim == 3 and array.shape[-1] == 1:
                array = array.squeeze(-1)   # (H, W, 1) --> (H, W)

            if array.ndim == 2:
                image_mode = "L"
            elif array.ndim == 3 and array.shape[-1] == 3:
                image_mode = "RGB"
            elif array.ndim == 3 and array.shape[-1] == 4:
                image_mode = "RGBA"
            else:
                raise ValueError(
                    f"[Save Temporary Image Named] Unsupported image shape: {array.shape}"
                )

            # Image.fromarray(array, mode="L").save(output_dir, format="PNG", compress_level=4)
            Image.fromarray(array, mode=image_mode).save(
                full_path,
                format="PNG",
                compress_level=4,
            )
            saved.append(full_path)

        return saved

    @staticmethod
    def _normalize_images(image):
        if not isinstance(image, torch.Tensor):
            raise TypeError("image must be a torch.Tensor")
        # if image.ndim != 4 or image.shape[-1] < 3:
        if image.ndim != 4 or image.shape[-1] not in (1, 3, 4):
            raise ValueError(
                "[Save Temporary Image Named] IMAGE must have shape "
                "[B,H,W,1], [B,H,W,3], or [B,H,W,4]."
            )
        return image

    @staticmethod
    def _resolve_output_directory(input_dir, output_folder):
        if os.path.isabs(output_folder):
            raise ValueError("output_folder must be relative to the ComfyUI input directory")

        input_root = os.path.abspath(input_dir)
        output_dir = os.path.abspath(os.path.join(input_root, output_folder))
        if os.path.commonpath([input_root, output_dir]) != input_root:
            raise ValueError("output_folder must remain inside the ComfyUI input directory")

        return output_dir
