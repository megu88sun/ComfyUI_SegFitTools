# ==============================================================================
# File Name:    densepose_crop_geometry.py
# Description:  Provides shared DensePose upper/lower crop geometry.
# Proposer:     MeguPaPa3
# Author:       OpenAI Codex
# Created:      2026-10-13
#
# Change Log:
# Date        Author        Description
# ----------  ------------  --------------------------------------------------
# 2026-10-13  OpenAI Codex  Extracted public crop geometry from Body Line Cropper.
# ==============================================================================

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch


class DensePoseCropGeometry:
    """Locate upper and lower body rectangles from a DensePose hint image."""

    @staticmethod
    def rgb8(image: torch.Tensor) -> np.ndarray:
        """Convert one ComfyUI IMAGE [H,W,C] to RGB uint8."""
        return (
            image[..., :3].detach().cpu().numpy().clip(0.0, 1.0)
            .__mul__(255.0).round().astype(np.uint8)
        )

    @staticmethod
    def _corner_background(rgb: np.ndarray) -> tuple[int, int, int]:
        """Determine the flat DensePose background color from image corners."""
        height, width = rgb.shape[:2]
        corners = np.array(
            [rgb[0, 0], rgb[0, width - 1], rgb[height - 1, 0],
             rgb[height - 1, width - 1]], dtype=np.uint8)
        colors, counts = np.unique(corners, axis=0, return_counts=True)
        color = colors[int(np.argmax(counts))]
        return tuple(int(value) for value in color)

    @staticmethod
    def _color_mask(rgb: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
        return np.all(rgb == np.asarray(color, dtype=np.uint8), axis=2)

    @classmethod
    def _find_torso(
        cls, rgb: np.ndarray, background: tuple[int, int, int]
    ) -> tuple[np.ndarray, tuple[int, int, int]]:
        """Select the largest non-background center-crossing torso region."""
        height, width = rgb.shape[:2]
        center_x = width // 2
        colors, counts = np.unique(rgb.reshape(-1, 3), axis=0, return_counts=True)
        candidates: list[tuple[int, tuple[int, int, int], np.ndarray]] = []

        for color_array, count in zip(colors, counts):
            color = tuple(int(value) for value in color_array)
            if color == background:
                continue
            mask = cls._color_mask(rgb, color)
            rows, columns = np.where(mask)
            if columns.size == 0:
                continue
            crosses_center = np.any(mask[:, max(0, center_x - 2):center_x + 3])
            vertical_span = int(rows.max() - rows.min() + 1)
            if crosses_center and vertical_span >= int(height * 0.20):
                candidates.append((int(count), color, mask))

        if not candidates:
            raise ValueError(
                "[DensePoseCropGeometry] Unable to identify torso color region."
            )
        _, torso_color, torso_mask = max(candidates, key=lambda item: item[0])
        return torso_mask, torso_color

    @staticmethod
    def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
        rows, columns = np.where(mask)
        if columns.size == 0:
            raise ValueError("[DensePoseCropGeometry] Empty mask has no bbox.")
        return (
            int(columns.min()), int(rows.min()),
            int(columns.max()), int(rows.max()),
        )

    @classmethod
    def _find_upper_arm_masks(
        cls,
        rgb: np.ndarray,
        background: tuple[int, int, int],
        torso_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find left and right upper-arm regions from lateral components."""
        height, width = rgb.shape[:2]
        torso_x1, torso_y1, torso_x2, torso_y2 = cls._bbox(torso_mask)
        torso_height = max(1, torso_y2 - torso_y1 + 1)
        torso_width = max(1, torso_x2 - torso_x1 + 1)
        center_x = (torso_x1 + torso_x2) / 2.0
        left_limit = torso_x1 + torso_width * 0.15
        right_limit = torso_x2 - torso_width * 0.15
        left = np.zeros((height, width), dtype=bool)
        right = np.zeros((height, width), dtype=bool)

        for color_array in np.unique(rgb.reshape(-1, 3), axis=0):
            color = tuple(int(value) for value in color_array)
            if color == background:
                continue
            color_mask = cls._color_mask(rgb, color)
            if np.array_equal(color_mask, torso_mask):
                continue

            # DensePose can assign one color to both arms. Classify disconnected
            # components independently so the two sides cannot collapse together.
            count, components = cv2.connectedComponents(
                color_mask.astype(np.uint8), connectivity=8)
            for component_id in range(1, count):
                mask = components == component_id
                rows, columns = np.where(mask)
                if columns.size == 0:
                    continue
                y_min = int(rows.min())
                y_max = int(rows.max())
                x_mean = float(columns.mean())
                area = int(columns.size)
                starts_near_shoulder = (
                    torso_y1 - int(torso_height * 0.03)
                    <= y_min <= torso_y1 + int(torso_height * 0.16)
                )
                ends_in_upper_body = y_max <= torso_y1 + int(torso_height * 0.58)
                large_enough = area >= max(64, int(height * width * 0.001))
                if not (starts_near_shoulder and ends_in_upper_body and large_enough):
                    continue
                if x_mean < center_x and x_mean <= left_limit:
                    left |= mask
                elif x_mean > center_x and x_mean >= right_limit:
                    right |= mask

        if not np.any(left) or not np.any(right):
            raise ValueError(
                "[DensePoseCropGeometry] Unable to identify both upper arms."
            )
        return left, right

    @classmethod
    def locate(
        cls,
        rgb: np.ndarray,
        upper_bottom_margin: float,
        lower_side_margin: float,
    ) -> dict[str, Any]:
        """Return standard crop_data geometry for one RGB DensePose image."""
        height, width = rgb.shape[:2]
        background = cls._corner_background(rgb)
        torso_mask, torso_color = cls._find_torso(rgb, background)
        left_arm, right_arm = cls._find_upper_arm_masks(
            rgb, background, torso_mask)
        torso_x1, torso_top, torso_x2, torso_bottom = cls._bbox(torso_mask)
        torso_height = max(1, torso_bottom - torso_top + 1)
        left_x1, _, left_x2, left_bottom = cls._bbox(left_arm)
        right_x1, _, right_x2, right_bottom = cls._bbox(right_arm)

        # Upper X bounds pass through the arm centers to retain garment sides
        # while excluding most arm-generated evidence lines.
        upper_x1 = int(round((left_x1 + left_x2) / 2.0))
        upper_x2 = int(round((right_x1 + right_x2) / 2.0))
        upper_y1 = torso_top
        upper_y2 = max(left_bottom, right_bottom) + 1
        upper_y2 += int(round(torso_height * upper_bottom_margin))

        # Lower detection begins at the shared upper boundary and extends over
        # the widest remaining torso area with a proportional side margin.
        lower_y1 = upper_y2
        lower_y2 = torso_bottom + 1
        lower_torso = torso_mask[upper_y2:torso_bottom + 1]
        _rows, columns = np.where(lower_torso)
        if columns.size > 0:
            lower_left = int(columns.min())
            lower_right = int(columns.max())
        else:
            lower_left = torso_x1
            lower_right = torso_x2
        lower_width = max(1, lower_right - lower_left + 1)
        side_margin = int(round(lower_width * lower_side_margin))
        lower_x1 = lower_left - side_margin
        lower_x2 = lower_right + side_margin + 1

        upper_x1 = int(np.clip(upper_x1, 0, width - 1))
        upper_x2 = int(np.clip(upper_x2, upper_x1 + 1, width))
        upper_y1 = int(np.clip(upper_y1, 0, height - 1))
        upper_y2 = int(np.clip(upper_y2, upper_y1 + 1, height))
        lower_x1 = int(np.clip(lower_x1, 0, width - 1))
        lower_x2 = int(np.clip(lower_x2, lower_x1 + 1, width))
        lower_y1 = int(np.clip(lower_y1, 0, height - 1))
        lower_y2 = int(np.clip(lower_y2, lower_y1 + 1, height))

        return {
            "image": {"width": width, "height": height},
            "analysis": {
                "background_rgb": list(background),
                "torso_rgb": list(torso_color),
                "torso_center_x": (torso_x1 + torso_x2) / 2.0,
                "torso_top": torso_top,
                "torso_bottom": torso_bottom,
                "left_upper_arm_bottom": left_bottom,
                "right_upper_arm_bottom": right_bottom,
            },
            "upper_crop": {
                "x1": upper_x1, "y1": upper_y1,
                "x2": upper_x2, "y2": upper_y2,
            },
            "lower_crop": {
                "x1": lower_x1, "y1": lower_y1,
                "x2": lower_x2, "y2": lower_y2,
            },
        }

    @staticmethod
    def crop_image(
        image: torch.Tensor, batch_index: int, box: dict[str, int]
    ) -> torch.Tensor:
        """Crop one IMAGE batch member with exclusive rectangle coordinates."""
        return image[
            batch_index, box["y1"]:box["y2"], box["x1"]:box["x2"], :
        ]
