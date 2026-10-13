# ==============================================================================
# File Name:    mask_refiner.py
# Description:  Extracts and refines tops, pants, and full-body underwear masks.
# Proposer:     MeguPaPa3
# Author:       OpenAI Codex
# Created:      2026-07-15
#
# Change Log:
# Date        Author        Description
# ----------  ------------  --------------------------------------------------
# 2026-07-15  OpenAI Codex  Initial release.
# 2026-07-18  OpenAI Codex  Integrated local SegFormer inference and full-body underwear output.
# ==============================================================================
"""SegFormer clothes segmentation with focused mask cleanup."""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F


_PROCESSOR = None
_MODEL = None


def _load_segmenter():
    global _PROCESSOR, _MODEL
    if _PROCESSOR is None or _MODEL is None:
        import folder_paths
        from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor

        model_path = os.path.realpath(
            os.path.join(folder_paths.models_dir, "segformer_b2_clothes")
        )
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"SegFormer model folder was not found: {model_path}"
            )
        _PROCESSOR = SegformerImageProcessor.from_pretrained(
            model_path, local_files_only=True
        )
        _MODEL = AutoModelForSemanticSegmentation.from_pretrained(
            model_path, local_files_only=True
        ).eval()
    return _PROCESSOR, _MODEL


class SegFormerMaskRefiner:
    """Extract and clean TOPS/PANTS masks directly from a person image."""

    CATEGORY = "SegFitTools/Mask"
    FUNCTION = "refine"
    RETURN_TYPES = ("MASK", "MASK", "IMAGE", "MASK")
    RETURN_NAMES = (
        "tops_mask",
        "pants_mask",
        "mask_preview",
        "fullbody_underwear_mask",
    )

    TOPS_LABEL = 4
    PANTS_LABEL = 6

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "remove_islands_smaller_than": (
                    "INT", {"default": 80, "min": 0, "max": 1000000}
                ),
                "fill_holes_smaller_than": (
                    "INT", {"default": 80, "min": 0, "max": 1000000}
                ),
                "grow_shrink_pixels": (
                    "INT", {"default": 0, "min": -128, "max": 128}
                ),
                "feather_pixels": (
                    "FLOAT", {"default": 1.0, "min": 0.0, "max": 64.0, "step": 0.1}
                ),
                "debug_mode": (["standard", "detailed"], {"default": "standard"}),
            },
            "optional": {
                "profile": ("SEGMASK_PROFILE",),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, image=None, **_kwargs):
        if image is not None and (
            not isinstance(image, torch.Tensor)
            or image.ndim != 4
            or image.shape[-1] < 3
        ):
            return "[SegFormer Mask Refiner] image must be an IMAGE tensor (B,H,W,C)."
        return True

    @staticmethod
    def _morph(mask, pixels):
        if pixels == 0:
            return mask
        radius = abs(int(pixels))
        value = mask.unsqueeze(1)
        if pixels > 0:
            value = F.max_pool2d(value, 2 * radius + 1, 1, radius)
        else:
            value = 1.0 - F.max_pool2d(
                1.0 - value, 2 * radius + 1, 1, radius
            )
        return value.squeeze(1)

    @staticmethod
    def _remove_islands(binary, minimum_area, ndimage):
        if minimum_area <= 0:
            return binary
        labels, count = ndimage.label(binary)
        if count == 0:
            return binary
        areas = np.bincount(labels.ravel())
        keep = areas >= minimum_area
        keep[0] = False
        return keep[labels]

    @staticmethod
    def _fill_holes(binary, maximum_area, ndimage):
        if maximum_area <= 0:
            return binary
        labels, count = ndimage.label(~binary)
        if count == 0:
            return binary
        border = np.unique(
            np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
        )
        areas = np.bincount(labels.ravel())
        fill = areas <= maximum_area
        fill[0] = False
        fill[border] = False
        return binary | fill[labels]

    @staticmethod
    def _profile_cleanup(profile, fill_holes, feather):
        if not profile:
            return fill_holes, feather
        opacity = profile.get("visual_opacity", "opaque_or_lined")
        lace = profile.get("lace_amount", "partial")
        lining = profile.get("lining", "partial")
        if opacity == "sheer":
            return 0, max(feather, 1.75)
        if opacity == "semi_sheer":
            return 0, max(feather, 1.25)
        if lining == "full":
            return fill_holes, feather
        if lining == "partial":
            return min(fill_holes, 60 if lace == "full" else 80), feather
        if lace == "full":
            return min(fill_holes, 20), feather
        if lace == "partial":
            return min(fill_holes, 40), feather
        return fill_holes, feather

    @staticmethod
    def _segment_image(image):
        from PIL import Image

        processor, model = _load_segmenter()
        array = (
            image[..., :3].detach().cpu().clamp(0, 1).numpy() * 255.0
        ).round().astype(np.uint8)
        pil_image = Image.fromarray(array, mode="RGB")
        inputs = processor(images=pil_image, return_tensors="pt")
        logits = model(**inputs).logits.detach().cpu()
        logits = F.interpolate(
            logits,
            size=(pil_image.height, pil_image.width),
            mode="bilinear",
            align_corners=False,
        )
        return logits.argmax(dim=1)[0].numpy()

    def _clean(self, binary, remove_islands, fill_holes, grow_shrink, feather, ndimage):
        value = self._morph(binary.float(), int(grow_shrink))
        cleaned = []
        for item in value.detach().cpu().numpy().astype(bool):
            item = self._remove_islands(item, int(remove_islands), ndimage)
            item = self._fill_holes(item, int(fill_holes), ndimage)
            item = item.astype(np.float32)
            if feather > 0:
                item = np.clip(
                    ndimage.gaussian_filter(item, sigma=float(feather) / 2.0),
                    0,
                    1,
                )
            cleaned.append(item)
        return torch.from_numpy(np.stack(cleaned)).to(
            device=binary.device, dtype=torch.float32
        )

    def refine(self, image, remove_islands_smaller_than,
               fill_holes_smaller_than, grow_shrink_pixels, feather_pixels,
               debug_mode, profile=None):
        try:
            from scipy import ndimage

            if image.ndim != 4 or image.shape[-1] < 3:
                raise ValueError("image must have shape (batch, height, width, channels).")
            if not torch.isfinite(image).all():
                raise ValueError("image contains NaN or infinite values.")

            fill_holes_smaller_than, feather_pixels = self._profile_cleanup(
                profile, int(fill_holes_smaller_than), float(feather_pixels)
            )
            segmentations = [self._segment_image(item) for item in image]
            labels = torch.from_numpy(np.stack(segmentations)).to(image.device)
            tops = self._clean(
                labels == self.TOPS_LABEL,
                remove_islands_smaller_than,
                fill_holes_smaller_than,
                grow_shrink_pixels,
                feather_pixels,
                ndimage,
            )
            pants = self._clean(
                labels == self.PANTS_LABEL,
                remove_islands_smaller_than,
                fill_holes_smaller_than,
                grow_shrink_pixels,
                feather_pixels,
                ndimage,
            )
            preview = torch.zeros(
                (*tops.shape, 3), device=image.device, dtype=torch.float32
            )
            preview[..., 0] = tops
            preview[..., 2] = pants
            fullbody_underwear = torch.maximum(tops, pants)
            return tops, pants, preview, fullbody_underwear
        except Exception as exc:
            if debug_mode == "detailed":
                raise
            raise RuntimeError(
                f"[SegFormer Mask Refiner] Segmentation or cleanup failed: {exc}"
            ) from None
