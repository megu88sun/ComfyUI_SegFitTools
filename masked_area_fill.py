# ==============================================================================
# File Name:    masked_area_fill.py
# Description:  Fills a masked IMAGE area with a solid RGB color.
# Proposer:     MeguPaPa3
# Author:       OpenAI Codex
# Created:      2026-07-19
#
# Change Log:
# Date        Author        Description
# ----------  ------------  --------------------------------------------------
# 2026-07-19  OpenAI Codex  Initial release.
# ==============================================================================

from __future__ import annotations

import torch
import torch.nn.functional as F


# ------------------------------------------------------------------------------
# Replaces only the masked RGB pixels with a user-selected solid color.
# Soft mask values blend the color with the source image, while unmasked pixels
# and any channels after RGB remain unchanged.
# ------------------------------------------------------------------------------
class MaskedAreaFill:
    CATEGORY = "SegFitTools/Image"
    FUNCTION = "fill"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "red": ("INT", {"default": 0, "min": 0, "max": 255}),
                "green": ("INT", {"default": 0, "min": 0, "max": 255}),
                "blue": ("INT", {"default": 0, "min": 0, "max": 255}),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, image=None, mask=None, **_kwargs):
        if image is not None and (
            not isinstance(image, torch.Tensor)
            or image.ndim != 4
            or image.shape[-1] < 3
        ):
            return "[Masked Area Fill] image must be an IMAGE tensor (B,H,W,C)."
        if mask is not None and (
            not isinstance(mask, torch.Tensor) or mask.ndim not in (2, 3)
        ):
            return "[Masked Area Fill] mask must be a MASK tensor (H,W) or (B,H,W)."
        return True

    @staticmethod
    def fill(image, mask, red, green, blue):
        if image.ndim != 4 or image.shape[-1] < 3:
            raise ValueError("[Masked Area Fill] Invalid IMAGE tensor shape.")
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim != 3:
            raise ValueError("[Masked Area Fill] Invalid MASK tensor shape.")
        if mask.shape[0] not in (1, image.shape[0]):
            raise ValueError(
                "[Masked Area Fill] MASK batch must be one or match the IMAGE batch."
            )
        if not torch.isfinite(image).all() or not torch.isfinite(mask).all():
            raise ValueError("[Masked Area Fill] IMAGE and MASK must contain finite values.")

        value = mask.to(device=image.device, dtype=image.dtype).clamp(0, 1)
        if value.shape[1:3] != image.shape[1:3]:
            value = F.interpolate(
                value.unsqueeze(1),
                size=image.shape[1:3],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        if value.shape[0] == 1 and image.shape[0] > 1:
            value = value.expand(image.shape[0], -1, -1)

        output = image.clone()
        alpha = value.unsqueeze(-1)
        color = torch.tensor(
            [red, green, blue], device=image.device, dtype=image.dtype
        ).view(1, 1, 1, 3) / 255.0
        output[..., :3] = output[..., :3] * (1.0 - alpha) + color * alpha
        return (output,)
