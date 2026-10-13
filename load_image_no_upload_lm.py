# ==============================================================================
# File Name:    load_image_no_upload_lm.py
# Description:  Loads and downscales large input images for low-profile systems.
# Proposer:     MeguPaPa3
# Author:       OpenAI Codex
# Created:      2026-07-27
#
# Change Log:
# Date        Author        Description
# ----------  ------------  --------------------------------------------------
# 2026-07-27  OpenAI Codex  Initial release.
# ==============================================================================

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

from .load_image_no_upload import LoadImageNoUpload


MAX_IMAGE_SIDE = 2000


# Provides the same interface as Load Image No Upload, but reduces oversized
# frames on the CPU before creating tensors or transferring them to a device.
class LoadImageNoUploadLM(LoadImageNoUpload):
    @staticmethod
    def _decode_image(image_path):
        source = Image.open(image_path)
        output_images = []
        output_masks = []
        width = None
        height = None

        for frame in ImageSequence.Iterator(source):
            frame = ImageOps.exif_transpose(frame)
            source_width, source_height = frame.size
            longest_side = max(source_width, source_height)

            if longest_side > MAX_IMAGE_SIDE:
                scale = MAX_IMAGE_SIDE / longest_side
                resized_width = max(1, round(source_width * scale))
                resized_height = max(1, round(source_height * scale))
                frame = frame.resize(
                    (resized_width, resized_height),
                    Image.Resampling.LANCZOS,
                )

            rgb = frame.convert("RGB")
            if width is None:
                width, height = rgb.size
            if rgb.size != (width, height):
                continue

            image_array = np.asarray(rgb, dtype=np.float32) / 255.0
            output_images.append(torch.from_numpy(image_array).unsqueeze(0))

            if "A" in frame.getbands():
                alpha = np.asarray(
                    frame.getchannel("A"),
                    dtype=np.float32,
                ) / 255.0
                mask = 1.0 - torch.from_numpy(alpha)
            else:
                mask = torch.zeros((height, width), dtype=torch.float32)
            output_masks.append(mask.unsqueeze(0))

        if not output_images:
            raise ValueError(f"No readable image frames found: {image_path}")

        return torch.cat(output_images, dim=0), torch.cat(output_masks, dim=0)
