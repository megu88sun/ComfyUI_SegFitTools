# ==============================================================================
# File Name:    advanced_fitter.py
# Description:  Provides advanced garment fitting and review gate nodes.
# Proposer:     MeguPaPa3
# Author:       OpenAI Codex
# Created:      2026-07-15
#
# Change Log:
# Date        Author        Description
# ----------  ------------  --------------------------------------------------
# 2026-07-15  OpenAI Codex  Initial release.
# ==============================================================================
"""Advanced mask-guided garment placement with review-friendly diagnostics."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .garment_fitter import FitGarmentToTargetMask


class FitGarmentAdvanced(FitGarmentToTargetMask):
    CATEGORY = "SegFitTools/Fit"
    FUNCTION = "fit_advanced"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = (
        "composite", "debug_preview", "garment_layer", "placed_mask",
        "garment_crop", "x", "y", "width", "height",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_image": ("IMAGE",),
                "source_image": ("IMAGE",),
                "source_mask": ("MASK",),
                "target_mask": ("MASK",),
                "garment_type": (["bra", "panties", "generic"], {"default": "bra"}),
                "fit_mode": (["width", "height", "contain", "stretch"], {"default": "width"}),
                "scale_x": ("FLOAT", {"default": 1.0, "min": 0.10, "max": 3.0, "step": 0.01}),
                "scale_y": ("FLOAT", {"default": 1.0, "min": 0.10, "max": 3.0, "step": 0.01}),
                "rotation": ("FLOAT", {"default": 0.0, "min": -45.0, "max": 45.0, "step": 0.5}),
                "offset_x": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "offset_y": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "mask_threshold": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask_grow": ("INT", {"default": 0, "min": -64, "max": 64}),
                "mask_blur": ("INT", {"default": 1, "min": 0, "max": 64}),
                "guide_opacity": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    @staticmethod
    def _anchor_from_mask(mask, threshold, mode):
        points = torch.nonzero(mask >= threshold, as_tuple=False)
        if points.numel() == 0:
            raise ValueError("A supplied mask is empty after thresholding.")
        ys = points[:, 0]
        xs = points[:, 1]
        if mode in ("underband", "crotch"):
            y0 = int(ys.min())
            y1 = int(ys.max())
            band_start = y1 - max(1, round((y1 - y0 + 1) * 0.10))
            band = points[ys >= band_start]
            return float(band[:, 1].float().median()), float(band[:, 0].float().median())
        return float(xs.float().median()), float(ys.float().median())

    @staticmethod
    def _rotate_expanded(image, mask, angle):
        if abs(float(angle)) < 1e-6:
            return image, mask
        height, width = image.shape[:2]
        side = max(1, int(math.ceil(math.hypot(width, height))))
        pad_x0 = (side - width) // 2
        pad_x1 = side - width - pad_x0
        pad_y0 = (side - height) // 2
        pad_y1 = side - height - pad_y0
        img = F.pad(image.movedim(-1, 0)[None], (pad_x0, pad_x1, pad_y0, pad_y1))
        msk = F.pad(mask[None, None], (pad_x0, pad_x1, pad_y0, pad_y1))
        radians = math.radians(float(angle))
        c, s = math.cos(radians), math.sin(radians)
        theta = torch.tensor([[[c, -s, 0.0], [s, c, 0.0]]], device=image.device, dtype=image.dtype)
        grid = F.affine_grid(theta, img.shape, align_corners=False)
        img = F.grid_sample(img, grid, mode="bicubic", padding_mode="zeros", align_corners=False)
        msk = F.grid_sample(msk, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        result_image = img[0].movedim(0, -1).clamp(0, 1)
        result_mask = msk[0, 0].clamp(0, 1)
        x0, y0, x1, y1 = FitGarmentToTargetMask._bbox(result_mask, 0.001)
        return result_image[y0:y1, x0:x1], result_mask[y0:y1, x0:x1]

    @staticmethod
    def _outline(mask):
        value = mask[None, None]
        eroded = 1.0 - F.max_pool2d(1.0 - value, 5, 1, 2)
        return (value - eroded).clamp(0, 1)[0, 0]

    @staticmethod
    def _cross(image, x, y, color, radius=8):
        output = image
        height, width = output.shape[:2]
        x, y = int(round(x)), int(round(y))
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        if 0 <= y < height:
            output[y, x0:x1, :3] = color
        if 0 <= x < width:
            output[y0:y1, x, :3] = color
        return output

    def fit_advanced(self, target_image, source_image, source_mask, target_mask,
                     garment_type, fit_mode, scale_x, scale_y, rotation,
                     offset_x, offset_y, mask_threshold, mask_grow, mask_blur,
                     guide_opacity):
        try:
            if target_image.ndim != 4 or source_image.ndim != 4:
                raise ValueError("Images must have shape (B,H,W,C).")
            source_mask = self._as_mask(source_mask)
            target_mask = self._as_mask(target_mask)
            batch = max(target_image.shape[0], source_image.shape[0], source_mask.shape[0], target_mask.shape[0])
            composites, previews, layers, masks, crops, positions = [], [], [], [], [], []
            anchor_mode = {"bra": "underband", "panties": "crotch"}.get(garment_type, "center")

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
                    ratio = tw / sw if fit_mode == "width" else th / sh
                    if fit_mode == "contain":
                        ratio = min(tw / sw, th / sh)
                    nw, nh = max(1, round(sw * ratio)), max(1, round(sh * ratio))
                nw = max(1, round(nw * float(scale_x)))
                nh = max(1, round(nh * float(scale_y)))

                fitted = self._resize_image(source[sy0:sy1, sx0:sx1][None], nh, nw)[0]
                alpha = self._resize_mask(smask[sy0:sy1, sx0:sx1][None], nh, nw)[0]
                fitted, alpha = self._rotate_expanded(fitted, alpha, rotation)
                alpha = self._blur(self._grow(alpha, int(mask_grow)), int(mask_blur))
                nh, nw = alpha.shape

                target_ax, target_ay = self._anchor_from_mask(tmask, mask_threshold, anchor_mode)
                source_ax, source_ay = self._anchor_from_mask(alpha, min(mask_threshold, 0.25), anchor_mode)
                x = round(target_ax - source_ax) + int(offset_x)
                y = round(target_ay - source_ay) + int(offset_y)

                canvas_h, canvas_w = target.shape[:2]
                dx0, dy0 = max(0, x), max(0, y)
                dx1, dy1 = min(canvas_w, x + nw), min(canvas_h, y + nh)
                if dx1 <= dx0 or dy1 <= dy0:
                    raise ValueError("The fitted garment is outside the target canvas.")
                sx_start, sy_start = dx0 - x, dy0 - y
                sx_end, sy_end = sx_start + dx1 - dx0, sy_start + dy1 - dy0
                patch = fitted[sy_start:sy_end, sx_start:sx_end].to(target.device, target.dtype)
                patch_alpha = alpha[sy_start:sy_end, sx_start:sx_end].to(target.device, target.dtype)
                channels = min(target.shape[-1], patch.shape[-1])
                layer = torch.zeros_like(target)
                full_mask = torch.zeros((canvas_h, canvas_w), device=target.device, dtype=target.dtype)
                layer[dy0:dy1, dx0:dx1, :channels] = patch[..., :channels]
                full_mask[dy0:dy1, dx0:dx1] = patch_alpha
                composite = target.clone()
                a = patch_alpha[..., None]
                composite[dy0:dy1, dx0:dx1, :channels] = patch[..., :channels] * a + composite[dy0:dy1, dx0:dx1, :channels] * (1.0 - a)

                preview = composite.clone()
                opacity = float(guide_opacity)
                red = self._outline((tmask >= mask_threshold).float()).to(preview.device, preview.dtype)
                green = self._outline((full_mask >= 0.25).float()).to(preview.device, preview.dtype)
                colors = torch.zeros_like(preview[..., :3])
                colors[..., 0] = red
                colors[..., 1] = green
                guide = torch.maximum(red, green)[..., None] * opacity
                preview[..., :3] = preview[..., :3] * (1.0 - guide) + colors * guide
                preview = self._cross(preview, target_ax, target_ay, torch.tensor([1.0, 0.0, 0.0], device=preview.device, dtype=preview.dtype))
                preview = self._cross(preview, x + source_ax, y + source_ay, torch.tensor([0.0, 1.0, 0.0], device=preview.device, dtype=preview.dtype))

                composites.append(composite)
                previews.append(preview)
                layers.append(layer)
                masks.append(full_mask)
                crops.append(fitted)
                positions.append((x, y, nw, nh))

            if len(crops) > 1 and any(c.shape != crops[0].shape for c in crops[1:]):
                raise ValueError("Batched inputs produced different garment crop sizes. Process them one at a time.")
            x, y, width, height = positions[0]
            return (torch.stack(composites), torch.stack(previews), torch.stack(layers),
                    torch.stack(masks), torch.stack(crops), x, y, width, height)
        except Exception as exc:
            raise RuntimeError(f"[Fit Garment Advanced] Fitting failed: {exc}") from None


class FitReviewGate:
    CATEGORY = "SegFitTools/Fit"
    FUNCTION = "review"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("preview_image", "approved_image", "approved_mask")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preview_image": ("IMAGE",),
                "fitted_image": ("IMAGE",),
                "fitted_mask": ("MASK",),
                "action": (["preview", "continue"], {"default": "preview"}),
            }
        }

    def review(self, preview_image, fitted_image, fitted_mask, action):
        if action == "continue":
            return preview_image, fitted_image, fitted_mask
        try:
            from comfy_execution.graph_utils import ExecutionBlocker
        except ImportError:
            from comfy_execution.graph import ExecutionBlocker
        return preview_image, ExecutionBlocker(None), ExecutionBlocker(None)
