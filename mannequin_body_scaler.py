# ==============================================================================
# File Name:    mannequin_body_scaler.py
# Description:  Fits cropped mannequin images to Body Line Cropper geometry.
# Proposer:     MeguPaPa3
# Author:       OpenAI Codex
# Created:      2026-10-10
#
# Change Log:
# Date        Author        Description
# ----------  ------------  --------------------------------------------------
# 2026-10-10  OpenAI Codex  Initial release.
# 2026-10-10  OpenAI Codex  Accepted Mannequin Cropper image outputs.
# 2026-10-10  OpenAI Codex  Added optional alpha-based person compositing.
# 2026-10-10  OpenAI Codex  Added aspect-preserving cover scaling.
# 2026-10-10  OpenAI Codex  Aligned cover scaling to DensePose torso centers.
# ==============================================================================
#

from __future__ import annotations

import json
import math
from typing import Any

import torch
import torch.nn.functional as functional


class MannequinBodyScaler:
    """Resize mannequin regions into another Body Line Cropper coordinate set."""

    CATEGORY = "SegFitTools/Preprocess"
    FUNCTION = "fit_mannequin"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = (
        "upper_mannequin", "lower_mannequin", "full_mannequin", "composite",
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        return {
            "required": {
                "upper_image": ("IMAGE",),
                "lower_image": ("IMAGE",),
                "target_crop_data": ("STRING", {"forceInput": True}),
                "mannequin_crop_data": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "removed_image": ("IMAGE",),
            },
        }

    @staticmethod
    def _records(crop_data: str, name: str) -> list[dict[str, Any]]:
        """Decode one or more Body Line Cropper records."""
        try:
            payload = json.loads(crop_data)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"[MannequinBodyScaler] {name} is not valid JSON."
            ) from exc
        records = payload.get("batch") if isinstance(payload, dict) else None
        if records is None:
            records = [payload]
        if not records or not all(isinstance(record, dict) for record in records):
            raise ValueError(
                f"[MannequinBodyScaler] {name} contains no crop records."
            )
        return records

    @staticmethod
    def _image_size(record: dict[str, Any], name: str) -> tuple[int, int]:
        """Read the full image size stored by Body Line Cropper."""
        image = record.get("image", {})
        try:
            width = int(image["width"])
            height = int(image["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"[MannequinBodyScaler] {name} has no valid image size."
            ) from exc
        if width <= 0 or height <= 0:
            raise ValueError(
                f"[MannequinBodyScaler] {name} image size must be positive."
            )
        return width, height

    @staticmethod
    def _box(
        record: dict[str, Any], key: str, width: int, height: int, name: str
    ) -> tuple[int, int, int, int]:
        """Validate one exclusive crop rectangle."""
        data = record.get(key, {})
        try:
            x1 = int(data["x1"])
            y1 = int(data["y1"])
            x2 = int(data["x2"])
            y2 = int(data["y2"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"[MannequinBodyScaler] {name} has no valid {key}."
            ) from exc
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(
                f"[MannequinBodyScaler] {name} {key} is outside its image."
            )
        return x1, y1, x2, y2

    @staticmethod
    def _resize(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Resize one ComfyUI HWC image while retaining its device and dtype."""
        if image.shape[:2] == (height, width):
            return image.clone()
        source_dtype = image.dtype
        work = image.permute(2, 0, 1).unsqueeze(0).to(torch.float32)
        resized = functional.interpolate(
            work, size=(height, width), mode="bilinear", align_corners=False)
        return resized[0].permute(1, 2, 0).to(source_dtype)

    @classmethod
    def _resize_cover(
        cls,
        image: torch.Tensor,
        height: int,
        width: int,
        source_center_x: float | None = None,
        target_center_x: float | None = None,
    ) -> torch.Tensor:
        """Fill a target rectangle without distortion or uncovered edges."""
        source_height, source_width = image.shape[:2]
        scale = max(width / source_width, height / source_height)
        resized_height = max(height, int(math.ceil(source_height * scale)))
        resized_width = max(width, int(math.ceil(source_width * scale)))
        resized = cls._resize(image, resized_height, resized_width)

        # Align the DensePose torso centers when available. Clamp the crop so
        # the mannequin still covers the complete target rectangle.
        y1 = (resized_height - height) // 2
        x1 = (resized_width - width) // 2
        if source_center_x is not None and target_center_x is not None:
            resized_center_x = source_center_x * resized_width / source_width
            x1 = int(round(resized_center_x - target_center_x))
            x1 = max(0, min(x1, resized_width - width))
        return resized[y1:y1 + height, x1:x1 + width, :].clone()

    @staticmethod
    def _torso_center_x(
        record: dict[str, Any], crop_box: tuple[int, int, int, int]
    ) -> float:
        """Return the torso center relative to one crop rectangle."""
        x1, _, x2, _ = crop_box
        center_x = record.get("analysis", {}).get("torso_center_x")
        if not isinstance(center_x, (int, float)):
            center_x = (x1 + x2) / 2.0
        return float(center_x) - x1

    @classmethod
    def _fit_one(
        cls,
        upper_image: torch.Tensor,
        lower_image: torch.Tensor,
        target: dict[str, Any],
        source: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fit source upper/lower pixels into one target full-size layer."""
        target_width, target_height = cls._image_size(target, "target_crop_data")
        source_width, source_height = cls._image_size(
            source, "mannequin_crop_data")
        source_upper = cls._box(
            source, "upper_crop", source_width, source_height,
            "mannequin_crop_data")
        source_lower = cls._box(
            source, "lower_crop", source_width, source_height,
            "mannequin_crop_data")
        expected_upper = (
            source_upper[3] - source_upper[1],
            source_upper[2] - source_upper[0],
        )
        expected_lower = (
            source_lower[3] - source_lower[1],
            source_lower[2] - source_lower[0],
        )
        if upper_image.shape[:2] != expected_upper:
            raise ValueError(
                "[MannequinBodyScaler] upper_image size does not match "
                "mannequin_crop_data upper_crop."
            )
        if lower_image.shape[:2] != expected_lower:
            raise ValueError(
                "[MannequinBodyScaler] lower_image size does not match "
                "mannequin_crop_data lower_crop."
            )

        target_upper = cls._box(
            target, "upper_crop", target_width, target_height, "target_crop_data")
        target_lower = cls._box(
            target, "lower_crop", target_width, target_height, "target_crop_data")
        # Build a target-sized lower layer. Pixels outside the two fitted Body
        # Line regions remain black and are hidden by the upper person layer.
        full = torch.zeros(
            (target_height, target_width, upper_image.shape[-1]),
            dtype=upper_image.dtype,
            device=upper_image.device,
        )
        fitted = []
        for source_crop, source_box, target_box in (
            (upper_image, source_upper, target_upper),
            (lower_image, source_lower, target_lower),
        ):
            tx1, ty1, tx2, ty2 = target_box
            source_center_x = cls._torso_center_x(source, source_box)
            target_center_x = cls._torso_center_x(target, target_box)
            target_crop = cls._resize_cover(
                source_crop,
                ty2 - ty1,
                tx2 - tx1,
                source_center_x,
                target_center_x,
            )
            full[ty1:ty2, tx1:tx2, :] = target_crop
            fitted.append(target_crop)
        return fitted[0], fitted[1], full

    @staticmethod
    def _stack(images: list[torch.Tensor], name: str) -> torch.Tensor:
        """Create a ComfyUI batch only when every fitted image has one shape."""
        shapes = {tuple(image.shape) for image in images}
        if len(shapes) != 1:
            raise ValueError(
                f"[MannequinBodyScaler] Batch produced different {name} sizes."
            )
        return torch.stack(images)

    @staticmethod
    def _composite_removed_image(
        full_mannequin: torch.Tensor, removed_image: torch.Tensor | None
    ) -> torch.Tensor:
        """Place an unchanged RGB foreground over mannequin using its alpha."""
        if removed_image is None:
            return full_mannequin[..., :3].clone()
        if not isinstance(removed_image, torch.Tensor):
            raise TypeError(
                "[MannequinBodyScaler] removed_image must be a torch.Tensor."
            )
        if removed_image.ndim != 4 or removed_image.shape[-1] < 4:
            raise ValueError(
                "[MannequinBodyScaler] removed_image must be RGBA [B,H,W,4]."
            )
        if removed_image.shape[0] not in (1, full_mannequin.shape[0]):
            raise ValueError(
                "[MannequinBodyScaler] removed_image batch must contain one "
                "image or match the target batch."
            )
        if removed_image.shape[1:3] != full_mannequin.shape[1:3]:
            raise ValueError(
                "[MannequinBodyScaler] removed_image must match the target canvas."
            )

        # Join Image With Alpha stores opacity in channel 4. Keep foreground
        # RGB pixels unchanged where alpha is one and reveal the mannequin only
        # through transparent or partially transparent underwear regions.
        foreground = removed_image[..., :3].to(
            device=full_mannequin.device, dtype=full_mannequin.dtype
        )
        alpha = removed_image[..., 3:4].to(
            device=full_mannequin.device, dtype=full_mannequin.dtype
        ).clamp(0.0, 1.0)
        if foreground.shape[0] == 1 and full_mannequin.shape[0] > 1:
            foreground = foreground.expand(full_mannequin.shape[0], -1, -1, -1)
            alpha = alpha.expand(full_mannequin.shape[0], -1, -1, -1)
        background = full_mannequin[..., :3]
        return foreground * alpha + background * (1.0 - alpha)

    @classmethod
    def fit_mannequin(
        cls,
        upper_image: torch.Tensor,
        lower_image: torch.Tensor,
        target_crop_data: str,
        mannequin_crop_data: str,
        removed_image: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return fitted upper, lower, and target-sized mannequin layers."""
        for name, image in (
            ("upper_image", upper_image), ("lower_image", lower_image),
        ):
            if not isinstance(image, torch.Tensor):
                raise TypeError(
                    f"[MannequinBodyScaler] {name} must be a torch.Tensor."
                )
            if image.ndim != 4 or image.shape[-1] < 3:
                raise ValueError(
                    f"[MannequinBodyScaler] {name} must be [B,H,W,C]."
                )
        if upper_image.shape[0] != lower_image.shape[0]:
            raise ValueError(
                "[MannequinBodyScaler] upper and lower image batches must match."
            )
        if upper_image.shape[-1] != lower_image.shape[-1]:
            raise ValueError(
                "[MannequinBodyScaler] upper and lower channel counts must match."
            )
        if upper_image.dtype != lower_image.dtype:
            raise ValueError(
                "[MannequinBodyScaler] upper and lower data types must match."
            )
        if upper_image.device != lower_image.device:
            raise ValueError(
                "[MannequinBodyScaler] upper and lower devices must match."
            )

        targets = cls._records(target_crop_data, "target_crop_data")
        sources = cls._records(mannequin_crop_data, "mannequin_crop_data")
        if len(sources) not in (1, len(targets)):
            raise ValueError(
                "[MannequinBodyScaler] mannequin crop batch must contain one "
                "record or match the target crop batch."
            )
        if upper_image.shape[0] not in (1, len(targets)):
            raise ValueError(
                "[MannequinBodyScaler] mannequin image batches must contain "
                "one image or match the target crop batch."
            )

        upper, lower, full = [], [], []
        for index, target in enumerate(targets):
            image_index = 0 if upper_image.shape[0] == 1 else index
            source_index = 0 if len(sources) == 1 else index
            fitted = cls._fit_one(
                upper_image[image_index], lower_image[image_index], target,
                sources[source_index])
            upper.append(fitted[0])
            lower.append(fitted[1])
            full.append(fitted[2])
        upper_batch = cls._stack(upper, "upper crop")
        lower_batch = cls._stack(lower, "lower crop")
        full_batch = cls._stack(full, "full layer")
        composite = cls._composite_removed_image(full_batch, removed_image)
        return upper_batch, lower_batch, full_batch, composite
