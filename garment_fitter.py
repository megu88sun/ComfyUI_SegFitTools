# ==============================================================================
# File Name:    garment_fitter.py
# Description:  Provides mask-guided garment fitting and compositing.
# Proposer:     MeguPaPa3
# Author:       OpenAI Codex
# Created:      2026-07-15
#
# Change Log:
# Date        Author        Description
# ----------  ------------  --------------------------------------------------
# 2026-07-15  OpenAI Codex  Initial release.
# ==============================================================================
"""Place a masked garment over a target region using mask bounding boxes."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class FitGarmentToTargetMask:
    CATEGORY = "SegFitTools/Fit"
    FUNCTION = "fit"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = (
        "composite", "garment_layer", "placed_mask", "garment_crop",
        "x", "y", "width", "height",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_image": ("IMAGE",),
                "source_image": ("IMAGE",),
                "source_mask": ("MASK",),
                "target_mask": ("MASK",),
                "fit_mode": (["width", "height", "contain", "stretch"], {"default": "width"}),
                "anchor": (["center", "top", "bottom"], {"default": "center"}),
                "scale_adjust": ("FLOAT", {"default": 1.0, "min": 0.10, "max": 3.0, "step": 0.01}),
                "offset_x": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "offset_y": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "mask_threshold": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask_grow": ("INT", {"default": 0, "min": -64, "max": 64}),
                "mask_blur": ("INT", {"default": 1, "min": 0, "max": 64}),
            }
        }

    @staticmethod
    def _as_mask(mask):
        if mask.ndim == 4:
            mask = mask[..., 0]
        if mask.ndim != 3:
            raise ValueError("Masks must have shape (B,H,W) or (B,H,W,C).")
        return mask.float().clamp(0, 1)

    @staticmethod
    def _bbox(mask, threshold):
        points = torch.nonzero(mask >= threshold, as_tuple=False)
        if points.numel() == 0:
            raise ValueError("A supplied mask is empty after thresholding.")
        y0, x0 = points.min(dim=0).values.tolist()
        y1, x1 = points.max(dim=0).values.tolist()
        return int(x0), int(y0), int(x1 + 1), int(y1 + 1)

    @staticmethod
    def _grow(mask, pixels):
        if pixels == 0:
            return mask
        radius = abs(int(pixels))
        value = mask[None, None]
        if pixels > 0:
            value = F.max_pool2d(value, 2 * radius + 1, 1, radius)
        else:
            value = 1.0 - F.max_pool2d(1.0 - value, 2 * radius + 1, 1, radius)
        return value[0, 0]

    @staticmethod
    def _blur(mask, radius):
        if radius <= 0:
            return mask
        size = int(radius) * 2 + 1
        kernel = torch.ones((1, 1, size, size), device=mask.device, dtype=mask.dtype) / (size * size)
        return F.conv2d(mask[None, None], kernel, padding=radius)[0, 0].clamp(0, 1)

    @staticmethod
    def _resize_image(image, height, width):
        return F.interpolate(image.movedim(-1, 1), (height, width), mode="bicubic", align_corners=False).movedim(1, -1).clamp(0, 1)

    @staticmethod
    def _resize_mask(mask, height, width):
        return F.interpolate(mask[:, None], (height, width), mode="nearest")[:, 0]

    def fit(self, target_image, source_image, source_mask, target_mask, fit_mode, anchor,
            scale_adjust, offset_x, offset_y, mask_threshold, mask_grow, mask_blur):
        try:
            if target_image.ndim != 4 or source_image.ndim != 4:
                raise ValueError("Images must have shape (B,H,W,C).")
            source_mask = self._as_mask(source_mask)
            target_mask = self._as_mask(target_mask)
            batch = max(target_image.shape[0], source_image.shape[0], source_mask.shape[0], target_mask.shape[0])

            composites, layers, masks, crops = [], [], [], []
            positions = []
            for index in range(batch):
                target = target_image[min(index, target_image.shape[0] - 1)]
                source = source_image[min(index, source_image.shape[0] - 1)]
                smask = source_mask[min(index, source_mask.shape[0] - 1)]
                tmask = target_mask[min(index, target_mask.shape[0] - 1)]
                sx0, sy0, sx1, sy1 = self._bbox(smask, mask_threshold)
                tx0, ty0, tx1, ty1 = self._bbox(tmask, mask_threshold)
                sw, sh, tw, th = sx1 - sx0, sy1 - sy0, tx1 - tx0, ty1 - ty0

                if fit_mode == "stretch":
                    nw, nh = tw, th
                else:
                    if fit_mode == "width":
                        scale = tw / sw
                    elif fit_mode == "height":
                        scale = th / sh
                    else:
                        scale = min(tw / sw, th / sh)
                    scale *= float(scale_adjust)
                    nw, nh = max(1, round(sw * scale)), max(1, round(sh * scale))
                if fit_mode == "stretch":
                    nw = max(1, round(nw * float(scale_adjust)))
                    nh = max(1, round(nh * float(scale_adjust)))

                crop_image = source[sy0:sy1, sx0:sx1][None]
                crop_mask = smask[sy0:sy1, sx0:sx1][None]
                fitted = self._resize_image(crop_image, nh, nw)[0]
                alpha = self._resize_mask(crop_mask, nh, nw)[0]
                alpha = self._blur(self._grow(alpha, int(mask_grow)), int(mask_blur))

                x = round(tx0 + (tw - nw) / 2) + int(offset_x)
                if anchor == "top":
                    y = ty0 + int(offset_y)
                elif anchor == "bottom":
                    y = ty1 - nh + int(offset_y)
                else:
                    y = round(ty0 + (th - nh) / 2) + int(offset_y)

                canvas_h, canvas_w = target.shape[0], target.shape[1]
                dx0, dy0 = max(0, x), max(0, y)
                dx1, dy1 = min(canvas_w, x + nw), min(canvas_h, y + nh)
                if dx1 <= dx0 or dy1 <= dy0:
                    raise ValueError("The fitted garment is outside the target canvas.")
                sx_start, sy_start = dx0 - x, dy0 - y
                sx_end, sy_end = sx_start + (dx1 - dx0), sy_start + (dy1 - dy0)

                layer = torch.zeros_like(target)
                full_mask = torch.zeros((canvas_h, canvas_w), device=target.device, dtype=target.dtype)
                patch = fitted[sy_start:sy_end, sx_start:sx_end].to(target.device, target.dtype)
                patch_alpha = alpha[sy_start:sy_end, sx_start:sx_end].to(target.device, target.dtype)
                channels = min(layer.shape[-1], patch.shape[-1])
                layer[dy0:dy1, dx0:dx1, :channels] = patch[..., :channels]
                full_mask[dy0:dy1, dx0:dx1] = patch_alpha
                composite = target.clone()
                a = patch_alpha[..., None]
                composite[dy0:dy1, dx0:dx1, :channels] = (
                    patch[..., :channels] * a + composite[dy0:dy1, dx0:dx1, :channels] * (1.0 - a)
                )

                composites.append(composite)
                layers.append(layer)
                masks.append(full_mask)
                crops.append(fitted)
                positions.append((x, y, nw, nh))

            # Variable crop sizes cannot form an IMAGE batch; single-image workflows are the intended use.
            if len(crops) > 1 and any(c.shape != crops[0].shape for c in crops[1:]):
                raise ValueError("Batched inputs produced different garment crop sizes. Process them one at a time.")
            x, y, width, height = positions[0]
            return (
                torch.stack(composites), torch.stack(layers), torch.stack(masks),
                torch.stack(crops), x, y, width, height,
            )
        except Exception as exc:
            raise RuntimeError(f"[Fit Garment to Target Mask] Fitting failed: {exc}") from None

