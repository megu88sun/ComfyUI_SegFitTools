# ==============================================================================
# File Name:    lingerie_splitter.py
# Description:  Splits a full lingerie mask into cropped bra and pants layers.
# Proposer:     MeguPaPa3
# Author:       OpenAI Codex
# Created:      2026-08-29
#
# Change Log:
# Date        Author        Description
# ----------  ------------  --------------------------------------------------
# 2026-08-29  OpenAI Codex  Initial release.
# 2026-10-13  OpenAI Codex  Replaced external part masks with automatic
#                           separation and transparent garment crops.
# ==============================================================================

from __future__ import annotations

import torch


class SplitLingeriePhoto:
    """Split one full lingerie mask into transparent bra and pants crops."""

    CATEGORY = "SegFitTools/Mask"
    FUNCTION = "split"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("bra_image", "pants_image", "bra_mask", "pants_mask")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "photo": ("IMAGE",),
                "full_mask": ("MASK",),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, photo=None, full_mask=None, **_kwargs):
        if photo is not None:
            if not isinstance(photo, torch.Tensor) or photo.ndim != 4:
                return "[Split Lingerie Photo] photo must be an IMAGE tensor (B,H,W,C)."
            if photo.shape[-1] < 3:
                return "[Split Lingerie Photo] photo must contain at least three channels."
        if full_mask is not None:
            if not isinstance(full_mask, torch.Tensor) or full_mask.ndim not in (3, 4):
                return "[Split Lingerie Photo] full_mask must be a MASK tensor (B,H,W)."
            if full_mask.ndim == 4 and full_mask.shape[-1] != 1:
                return "[Split Lingerie Photo] a four-dimensional full_mask must have one channel."
        return True

    @staticmethod
    def _normalise_mask(mask, batch, height, width, device):
        if mask.ndim == 4:
            if mask.shape[-1] != 1:
                raise ValueError("a four-dimensional full_mask must have one channel")
            mask = mask[..., 0]
        if mask.shape[-2:] != (height, width):
            raise ValueError("full_mask must match the photo height and width")
        if mask.shape[0] == 1 and batch > 1:
            mask = mask.expand(batch, -1, -1)
        elif mask.shape[0] != batch:
            raise ValueError("full_mask batch size must be 1 or match the photo batch size")
        if not torch.isfinite(mask).all():
            raise ValueError("full_mask contains NaN or infinite values")
        return mask.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)

    @staticmethod
    def _find_split_row(mask):
        # A small row-coverage threshold ignores isolated mask noise while retaining
        # thin straps and side sections of the garment.
        binary = mask >= 0.5
        row_counts = binary.sum(dim=1)
        minimum_row_pixels = max(1, int(round(mask.shape[1] * 0.002)))
        active_rows = torch.nonzero(row_counts >= minimum_row_pixels, as_tuple=False).flatten()
        if active_rows.numel() < 2:
            raise ValueError("full_mask does not contain enough garment pixels")

        # Select the widest empty vertical interval that leaves a meaningful
        # garment region on both sides.
        total_pixels = int(binary.sum().item())
        minimum_part_pixels = max(16, int(round(total_pixels * 0.05)))
        candidates = []
        for index in range(active_rows.numel() - 1):
            upper_row = int(active_rows[index].item())
            lower_row = int(active_rows[index + 1].item())
            gap = lower_row - upper_row - 1
            if gap <= 0:
                continue
            split_y = (upper_row + lower_row + 1) // 2
            upper_pixels = int(binary[:split_y].sum().item())
            lower_pixels = int(binary[split_y:].sum().item())
            if upper_pixels >= minimum_part_pixels and lower_pixels >= minimum_part_pixels:
                candidates.append((gap, split_y))
        if not candidates:
            raise ValueError("full_mask has no vertical gap separating bra and pants")
        return max(candidates, key=lambda item: item[0])[1]

    @staticmethod
    def _build_rgba(photo, mask):
        # The generated part mask is the authoritative alpha channel. This also
        # makes an RGBA input behave the same way as an RGB input.
        rgb = photo[..., :3].to(dtype=torch.float32).clamp(0.0, 1.0)
        alpha = mask.to(dtype=rgb.dtype).unsqueeze(-1)
        return torch.cat((rgb, alpha), dim=-1)

    @staticmethod
    def _batch_bbox(mask, label):
        # A union box keeps every item in a ComfyUI batch stackable while each
        # garment type remains independently cropped.
        active = (mask >= 0.5).any(dim=0)
        points = torch.nonzero(active, as_tuple=False)
        if points.numel() == 0:
            raise ValueError(f"{label} mask is empty")
        y0 = int(points[:, 0].min().item())
        y1 = int(points[:, 0].max().item()) + 1
        x0 = int(points[:, 1].min().item())
        x1 = int(points[:, 1].max().item()) + 1
        return x0, y0, x1, y1

    @classmethod
    def _crop_rgba(cls, photo, mask, box):
        x0, y0, x1, y1 = box
        image_crop = photo[:, y0:y1, x0:x1]
        mask_crop = mask[:, y0:y1, x0:x1]
        return cls._build_rgba(image_crop, mask_crop), mask_crop

    @classmethod
    def split(cls, photo, full_mask):
        try:
            if not isinstance(photo, torch.Tensor) or photo.ndim != 4:
                raise ValueError("photo must be an IMAGE tensor (B,H,W,C)")
            if photo.shape[-1] < 3:
                raise ValueError("photo must contain at least three channels")
            if not torch.isfinite(photo).all():
                raise ValueError("photo contains NaN or infinite values")

            batch, height, width, _channels = photo.shape
            full = cls._normalise_mask(full_mask, batch, height, width, photo.device)
            bra_masks = []
            pants_masks = []
            for item in full:
                split_y = cls._find_split_row(item)
                bra = item.clone()
                pants = item.clone()
                bra[split_y:] = 0.0
                pants[:split_y] = 0.0
                bra_masks.append(bra)
                pants_masks.append(pants)

            bra_mask = torch.stack(bra_masks, dim=0)
            pants_mask = torch.stack(pants_masks, dim=0)
            bra_box = cls._batch_bbox(bra_mask, "bra")
            pants_box = cls._batch_bbox(pants_mask, "pants")
            bra_image, bra_crop_mask = cls._crop_rgba(photo, bra_mask, bra_box)
            pants_image, pants_crop_mask = cls._crop_rgba(photo, pants_mask, pants_box)
            return bra_image, pants_image, bra_crop_mask, pants_crop_mask
        except Exception as exc:
            raise RuntimeError(f"[Split Lingerie Photo] Split failed: {exc}") from None
