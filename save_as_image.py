# ==============================================================================
# File Name:    save_as_image.py
# Description:  Saves images with source-based names and conflict rules.
# Proposer:     MeguPaPa3
# Author:       MeguPaPa3
# Created:      2026-07-21
#
# Change Log:
# Date        Author        Description
# ----------  ------------  --------------------------------------------------
# 2026-07-25  MeguPaPa3     Initial release.
# 2026-07-26  OpenAI Codex  Fixed output paths, image modes, and batch handling.
# 2026-07-30  MeguPaPa3     Change Name. ( save_temp_image_named -> save_as_image )
# 2026-07-30  OpenAI Codex  Fixed identifiers, messages, and argument order.
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

# Saves IMAGE tensors as PNG files under ComfyUI's input directory.
# overwrite: Replaces a file with the same source stem and identifier.
# increment: Appends a revision number while retaining the source association.
# error    : Stops if the destination file already exists.
class SaveAsImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "source_filename": ("STRING", {"default": "source.png"}),
                "output_folder": ("STRING", {"default": "SegFit_TEMP"}),
                # "identifer": ("STRING", {"default": "TEMP"}),
                "identifier": ("STRING", {"default": "TEMP"}),
                "on_conflict": (["overwrite", "increment", "error"], {"default": "overwrite"}),
            }
        }

    CATEGORY = "SegFitTools/Image"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("saved_filename", "saved_path")
    FUNCTION = "save_image"
    OUTPUT_NODE = True

    # def save_image(self, image, source_filename, identifer, output_folder, on_conflict):
    def save_image(self, image, source_filename, output_folder, identifier, on_conflict):
        import folder_paths

        input_dir = folder_paths.get_input_directory()
        # saved = self._save_images(image, input_dir, output_folder, source_filename, identifer, on_conflict)
        saved = self._save_images(image, input_dir, output_folder, source_filename, identifier, on_conflict)

        ui_images = []
        for path in saved:
            relative = os.path.relpath(path, input_dir)
            ui_images.append({
                "filename": os.path.basename(relative),
                "subfolder": os.path.dirname(relative).replace(os.sep, "/"),
                "type": "input", 
            })
        # endfor

        filenames = "\n".join(os.path.basename(path) for path in saved)
        paths = "\n".join(saved)
        return {"ui": {"images": ui_images}, "result": (filenames, paths)}

    @classmethod
    # def _save_images(cls, image, input_dir, output_folder, source_filename, identifer, on_conflict):
    def _save_images(cls, image, input_dir, output_folder, source_filename, identifier, on_conflict):
        output_dir = cls._resolve_output_directory(input_dir, output_folder)
        os.makedirs(output_dir, exist_ok=True)

        source_stem = _safe_name(os.path.splitext(os.path.basename(source_filename))[0], "source")
        images = cls._normalize_images(image)
        base_name = source_stem
        # if str(identifer).strip():
        #     identifer_str = _safe_name(identifer, "Image")
        #     base_name = f"{base_name}__{identifer_str}"
        if str(identifier).strip():
            identifier_str = _safe_name(identifier, "Image")
            base_name = f"{base_name}__{identifier_str}"
        # endif
    
        saved = []
        for batch_index, single_image in enumerate(images):
            item_name = base_name
            if len(images) > 1:
                item_name = f"{item_name}__batch{batch_index + 1:03d}"
            # endif

            full_path = cls._select_path(output_dir, item_name, on_conflict)
            array = (single_image.detach().float().clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)

            if array.ndim == 3 and array.shape[-1] == 1:
                array = array.squeeze(-1)   # (H, W, 1) --> (H, W)
            # endif

            if array.ndim == 2:
                image_mode = "L"            # Grayscale image
            elif array.ndim == 3 and array.shape[-1] == 3:
                image_mode = "RGB"
            elif array.ndim == 3 and array.shape[-1] == 4:
                image_mode = "RGBA"
            else:
                # raise ValueError(f"[Save Temporary Image Named] Unsupported image shape: {array.shape}")
                raise ValueError(f"[Save As Image] Unsupported image shape: {array.shape}")
            # endif
    
            Image.fromarray(array, mode=image_mode).save(full_path, format="PNG", compress_level=4)
            saved.append(full_path)
        # endfor
        return saved

    @staticmethod
    def _normalize_images(image):
        if not isinstance(image, torch.Tensor):
            raise TypeError("image must be a torch.Tensor")
        # endif
        if image.ndim != 4 or image.shape[-1] not in (1, 3, 4):
            # raise ValueError("[Save Temporary Image Named] IMAGE must have shape [B,H,W,1], [B,H,W,3], or [B,H,W,4].")
            raise ValueError("[Save As Image] IMAGE must have shape [B,H,W,1], [B,H,W,3], or [B,H,W,4].")
        # endif
        return image

    @staticmethod
    def _resolve_output_directory(input_dir, output_folder):
        if os.path.isabs(output_folder):
            raise ValueError("output_folder must be relative to the ComfyUI input directory")
        # endif
        # 
        # Ensure that the output folder is within the input directory to prevent saving files
        # outside of the intended area.
        input_root = os.path.abspath(input_dir)
        output_dir = os.path.abspath(os.path.join(input_root, output_folder))
        if os.path.commonpath([input_root, output_dir]) != input_root:
            raise ValueError("output_folder must remain inside the ComfyUI input directory")
        # endif
        return output_dir

    @staticmethod
    def _select_path(output_dir, base_name, on_conflict):
        base_path = os.path.join(output_dir, f"{base_name}.png")

        if on_conflict == "overwrite":
            return base_path
        # endif
        if on_conflict == "error":
            if os.path.exists(base_path):
                raise FileExistsError(f"Image already exists: {base_path}")
            # endif
            return base_path
        # endif
        if on_conflict == "increment":
            revision = 1
            while True:
                path = os.path.join(output_dir, f"{base_name}__{revision:03d}.png")
                if not os.path.exists(path):
                    return path
                # endif
                revision += 1
            # endwhile  
        # endif
    
        raise ValueError(f"Unknown on_conflict mode: {on_conflict}")
