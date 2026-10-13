# ==============================================================================
# File Name:    load_image_no_upload.py
# Description:  Loads an existing ComfyUI input image without an upload control.
# Proposer:     MeguPaPa3
# Author:       OpenAI Codex
# Created:      2026-07-21
#
# Change Log:
# Date        Author        Description
# ----------  ------------  --------------------------------------------------
# 2026-07-21  OpenAI Codex  Initial release.
# 2026-07-25  OpenAI Codex  Scoped image selection to a configurable input folder.
# 2026-07-26  OpenAI Codex  Fixed dynamic-folder validation and encoded paths.
# ==============================================================================

import hashlib
import html
import os

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

#DEFAULT_FOLDER_NAME = "SegFit_MODELS"
DEFAULT_FOLDER_NAME = ""

# Loads an image already present under ComfyUI's input directory, returns its
# inverted alpha mask and base filename, and delegates preview display to the
# standard ComfyUI image widget without exposing its upload control.
class LoadImageNoUpload:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        files = cls._list_images(
            folder_paths.get_input_directory(),
            DEFAULT_FOLDER_NAME,
            folder_paths.filter_files_content_types,
        )
        return {
            "required": {
                "folder_name": (
                    "STRING",
                    {"default": DEFAULT_FOLDER_NAME},
                ),
                "image": (
                    files or [""],
                    {"image_upload": True},
                ),
            }
        }

    CATEGORY = "SegFitTools/Image"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "filename", "filename_stem")
    FUNCTION = "load_image"

    @classmethod
    def VALIDATE_INPUTS(cls, folder_name, image):
        import folder_paths

        try:
            image_path = cls._resolve_image_path(
                folder_paths.get_input_directory(),
                folder_name,
                image,
            )
        except (TypeError, ValueError) as error:
            return str(error)
        if not os.path.isfile(image_path):
            return f"[Load Image No Upload] Not Found: {image_path}"
        return True

    def load_image(self, folder_name, image):
        import folder_paths

        image_path = self._resolve_image_path(
            folder_paths.get_input_directory(),
            folder_name,
            image,
        )
        if not os.path.isfile(image_path):
            raise FileNotFoundError(
                f"[Load Image No Upload] Not Found: {image_path}"
            )
        output_image, output_mask = self._decode_image(image_path)

        try:
            import comfy.model_management

            dtype = comfy.model_management.intermediate_dtype()
            device = comfy.model_management.intermediate_device()
            output_image = output_image.to(device=device, dtype=dtype)
            output_mask = output_mask.to(device=device, dtype=dtype)
        except (ImportError, AttributeError):
            pass

        filename = os.path.basename(image_path)
        filename_stem = os.path.splitext(filename)[0]
        return output_image, output_mask, filename, filename_stem

    @staticmethod
    def _resolve_folder(input_dir, folder_name):
        input_root = os.path.realpath(input_dir)
        folder_name = str(folder_name or DEFAULT_FOLDER_NAME).strip().strip('"')
        candidate = (
            folder_name
            if os.path.isabs(folder_name)
            else os.path.join(input_root, folder_name)
        )
        folder_path = os.path.realpath(candidate)
        try:
            if os.path.commonpath((input_root, folder_path)) != input_root:
                raise ValueError(
                    "[Load Image No Upload] folder_name must remain inside "
                    "the ComfyUI input directory."
                )
        except ValueError as error:
            raise ValueError(
                "[Load Image No Upload] folder_name must remain inside "
                "the ComfyUI input directory."
            ) from error
        return folder_path

    @classmethod
    def _list_images(cls, input_dir, folder_name, content_filter=None):
        folder_path = cls._resolve_folder(input_dir, folder_name)
        if not os.path.isdir(folder_path):
            return []

        files = []
        for root, _, names in os.walk(folder_path):
            for name in names:
                path = os.path.join(root, name)
                if os.path.isfile(path):
                    files.append(
                        os.path.relpath(path, input_dir).replace(os.sep, "/")
                    )
        if content_filter is not None:
            files = content_filter(files, ["image"])
        else:
            image_extensions = {
                ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"
            }
            files = [
                path
                for path in files
                if os.path.splitext(path)[1].lower() in image_extensions
            ]
        return sorted(files)

    @classmethod
    def _resolve_image_path(cls, input_dir, folder_name, image):
        folder_path = cls._resolve_folder(input_dir, folder_name)
        # image = str(image or "").strip().strip('"')
        image = cls._decode_widget_value(image).strip().strip('"')
        if not image:
            raise ValueError("[Load Image No Upload] No image is selected.")

        if image.endswith("[input]"):
            image = image[:-7].rstrip()
        candidate = os.path.realpath(os.path.join(input_dir, image))
        try:
            inside_folder = os.path.commonpath((folder_path, candidate)) == folder_path
        except ValueError:
            inside_folder = False
        if not inside_folder:
            candidate = os.path.realpath(os.path.join(folder_path, image))
        try:
            if os.path.commonpath((folder_path, candidate)) != folder_path:
                raise ValueError(
                    "[Load Image No Upload] image must remain inside folder_name."
                )
        except ValueError as error:
            raise ValueError(
                "[Load Image No Upload] image must remain inside folder_name."
            ) from error
        return candidate

    @staticmethod
    def _decode_widget_value(value):
        decoded = str(value or "")
        for _ in range(4):
            next_value = html.unescape(decoded)
            if next_value == decoded:
                break
            decoded = next_value
        return decoded

    @staticmethod
    def _decode_image(image_path):
        source = Image.open(image_path)
        output_images = []
        output_masks = []
        width = None
        height = None

        for frame in ImageSequence.Iterator(source):
            frame = ImageOps.exif_transpose(frame)
            rgb = frame.convert("RGB")

            if width is None:
                width, height = rgb.size
            if rgb.size != (width, height):
                continue

            image_array = np.asarray(rgb, dtype=np.float32) / 255.0
            output_images.append(torch.from_numpy(image_array).unsqueeze(0))

            if "A" in frame.getbands():
                alpha = np.asarray(frame.getchannel("A"), dtype=np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(alpha)
            else:
                mask = torch.zeros((height, width), dtype=torch.float32)
            output_masks.append(mask.unsqueeze(0))

        if not output_images:
            raise ValueError(f"No readable image frames found: {image_path}")

        return torch.cat(output_images, dim=0), torch.cat(output_masks, dim=0)

    @classmethod
    def IS_CHANGED(cls, folder_name, image):
        import folder_paths

        image_path = cls._resolve_image_path(
            folder_paths.get_input_directory(),
            folder_name,
            image,
        )
        digest = hashlib.sha256()
        with open(image_path, "rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()


# Provides folder-scoped image lists to the frontend without placing every
# image under ComfyUI input into the node definition.
try:
    from aiohttp import web
    import folder_paths
    from server import PromptServer

    if getattr(PromptServer, "instance", None) is not None:
        @PromptServer.instance.routes.get("/segfittools/input-images")
        async def segfittools_input_images(request):
            folder_name = request.query.get("folder_name", DEFAULT_FOLDER_NAME)
            try:
                files = LoadImageNoUpload._list_images(
                    folder_paths.get_input_directory(),
                    folder_name,
                    folder_paths.filter_files_content_types,
                )
            except ValueError as error:
                return web.json_response({"error": str(error)}, status=400)
            return web.json_response(files)
except ImportError:
    pass
