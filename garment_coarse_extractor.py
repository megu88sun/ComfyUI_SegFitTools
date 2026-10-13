# ==============================================================================
# File Name:    garment_coarse_extractor.py
# Description:  Robust garment region extractor designed for complex lingerie
#               structures (side-flare lace, inside-seam elastics, trapezoidal
#               front panels). Employs two-stage leveling (gray-leveling for
#               fiber/ribbon seams and solid black-flood inversion) with LAB
#               affinity and skin-tone exclusion. Supports boundary feathering.
# Proposer:     MeguPaPa3
# Author:       Google Gemini
# Created:      2026-09-18
# Updated:      2026-09-25
#
# Change Log:
# Date        Author         Description
# ----------  -------------  --------------------------------------------------
# 2026-09-18  Google Gemini  Initial implementation.
# 2026-09-24  Google Gemini  Added two-stage leveling and UI seed forwarding.
# 2026-09-25  Google Gemini  Fixed INPUT_TYPES dictionary structure and added
#                            boundary feathering (mask_expand, mask_blur) in
#                            strict compliance with AGENTS.md.
# ==============================================================================

from __future__ import annotations

import json
import os
from typing import Any
import cv2
import numpy as np
import torch

# The Rules of Import: Safe fallback for standalone test execution
try:
    import folder_paths
except ImportError:
    class MockFolderPaths:
        @staticmethod
        def get_input_directory() -> str:
            return os.path.abspath(os.path.join(os.path.dirname(__file__), "mock_input"))

    folder_paths = MockFolderPaths()


class GarmentCoarseExtractor:
    CATEGORY = "SegFitTools/Preprocess"
    FUNCTION = "extract_garment_regions"
    RETURN_TYPES = ("MASK", "MASK", "IMAGE")
    RETURN_NAMES = ("bra_mask", "brief_mask", "debug_map")

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        return {
            "required": {
                "image": ("IMAGE",),
                "seed_points_json": (
                    "STRING",
                    {
                        "default": '{"bra": [[0.5, 0.28]], "brief": [[0.5, 0.72]]}',
                        "multiline": True,
                    },
                ),
                "color_tolerance": (
                    "FLOAT",
                    {
                        "default": 32.0,
                        "min": 5.0,
                        "max": 80.0,
                        "step": 1.0,
                    },
                ),
                "lace_bridge_size": (
                    "INT",
                    {
                        "default": 13,
                        "min": 3,
                        "max": 41,
                        "step": 2,
                    },
                ),
                "exclude_skin": (
                    ["enable", "disable"],
                    {"default": "enable"},
                ),
            },
            "optional": {
                "mask_expand": (
                    "INT",
                    {
                        "default": 2,
                        "min": 0,
                        "max": 16,
                        "step": 1,
                    },
                ),
                "mask_blur": (
                    "INT",
                    {
                        "default": 3,
                        "min": 0,
                        "max": 31,
                        "step": 1,
                    },
                ),
            },
        }

    @staticmethod
    def _detect_skin_mask(bgr: np.ndarray) -> np.ndarray:
        """Detect human skin regions in YCrCb and HSV spaces."""
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        mask_ycrcb = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, (0, 15, 60), (25, 255, 255))

        skin = cv2.bitwise_and(mask_ycrcb, mask_hsv)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel)
        return skin

    @staticmethod
    def _level_fiber_transitions(bgr: np.ndarray) -> np.ndarray:
        """
        Stage 1: 'Gray-leveling' equivalent.
        Flattens high-frequency seam lines, lace meshes, and ribbon shadow grooves
        while preserving overall boundary contours using Edge-Preserving Bilateral Filtering.
        """
        smoothed_bgr = cv2.bilateralFilter(bgr, d=9, sigmaColor=75, sigmaSpace=75)
        return smoothed_bgr

    @staticmethod
    def _feather_mask(mask: np.ndarray, expand: int, blur: int) -> np.ndarray:
        """Slightly dilate to encompass lace fibers, then apply Gaussian blur for soft transition."""
        m = mask.copy()
        if expand > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand * 2 + 1, expand * 2 + 1))
            m = cv2.dilate(m, kernel, iterations=1)
        if blur > 0:
            ksize = blur * 2 + 1
            m = cv2.GaussianBlur(m, (ksize, ksize), 0)
        return m

    @classmethod
    def _grow_garment_region(
        cls,
        lab: np.ndarray,
        bgr: np.ndarray,
        seeds: list[tuple[int, int]],
        tolerance: float,
        bridge_size: int,
        skin_mask: np.ndarray | None,
    ) -> np.ndarray:
        """
        Grows garment regions with two-stage leveling:
        1. Gray-tone mesh leveling to bridge lace/ribbon seams.
        2. Solid hull closure and inversion against background.
        """
        h, w = lab.shape[:2]
        accum_mask = np.zeros((h, w), dtype=np.uint8)
        non_background = cv2.inRange(bgr, (5, 5, 5), (255, 255, 255))

        for sx, sy in seeds:
            sx = int(np.clip(sx, 0, w - 1))
            sy = int(np.clip(sy, 0, h - 1))

            y1, y2 = max(0, sy - 2), min(h, sy + 3)
            x1, x2 = max(0, sx - 2), min(w, sx + 3)
            patch = lab[y1:y2, x1:x2]
            seed_color = np.median(patch.reshape(-1, 3), axis=0)

            diff = np.linalg.norm(lab.astype(np.float32) - seed_color, axis=2)
            seed_match = (diff <= tolerance).astype(np.uint8) * 255
            seed_match[non_background == 0] = 0

            if skin_mask is not None:
                if skin_mask[sy, sx] == 0:
                    seed_match[skin_mask > 0] = 0

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(seed_match, connectivity=8)
            target_label = labels[sy, sx]

            if target_label > 0:
                component = (labels == target_label).astype(np.uint8) * 255
            else:
                patch_labels = labels[y1:y2, x1:x2]
                valid_labels = patch_labels[patch_labels > 0]
                if valid_labels.size > 0:
                    best_label = int(np.bincount(valid_labels).argmax())
                    component = (labels == best_label).astype(np.uint8) * 255
                else:
                    component = np.zeros((h, w), dtype=np.uint8)

            accum_mask = cv2.bitwise_or(accum_mask, component)

        if bridge_size > 1:
            k_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge_size, bridge_size))
            accum_mask = cv2.morphologyEx(accum_mask, cv2.MORPH_CLOSE, k_bridge)

            h_bridge_w = max(bridge_size * 2 + 1, 25)
            k_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (h_bridge_w, 3))
            accum_mask = cv2.morphologyEx(accum_mask, cv2.MORPH_CLOSE, k_horizontal)

            k_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            accum_mask = cv2.dilate(accum_mask, k_connect, iterations=1)

        contours, _ = cv2.findContours(accum_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(accum_mask)
        cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)

        if bridge_size > 1:
            filled_mask = cv2.erode(filled_mask, k_connect, iterations=1)

        filled_mask[non_background == 0] = 0
        return filled_mask

    @classmethod
    def extract_garment_regions(
        cls,
        image: torch.Tensor,
        seed_points_json: str,
        color_tolerance: float,
        lace_bridge_size: int,
        exclude_skin: str,
        mask_expand: int = 2,
        mask_blur: int = 3,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Executes coarse garment extraction with boundary feathering.
        Input image: [B, H, W, C] in 0.0 - 1.0 (float32).
        Outputs masks: [B, H, W] in 0.0 - 1.0 (float32).
        Outputs debug_map: [B, H, W, 3] in 0.0 - 1.0 (float32).
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError("[GarmentCoarseExtractor] 'image' must be a torch.Tensor")
        if image.ndim != 4 or image.shape[-1] < 3:
            raise ValueError("[GarmentCoarseExtractor] 'image' must have shape [B, H, W, 3] or [B, H, W, 4]")

        try:
            seed_data: dict[str, Any] = json.loads(seed_points_json)
            if not isinstance(seed_data, dict):
                seed_data = {}
        except Exception:
            seed_data = {}

        batch_size, h, w, _ = image.shape
        device = image.device
        dtype = image.dtype

        bra_masks: list[torch.Tensor] = []
        brief_masks: list[torch.Tensor] = []
        debug_maps: list[torch.Tensor] = []

        for b in range(batch_size):
            rgb = (image[b, ..., :3].detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB if cv2.__file__ is None else cv2.COLOR_RGB2BGR)

            leveled_bgr = cls._level_fiber_transitions(bgr)
            lab = cv2.cvtColor(leveled_bgr, cv2.COLOR_BGR2LAB)

            skin_mask = cls._detect_skin_mask(bgr) if exclude_skin == "enable" else None

            def parse_pts(key: str) -> list[tuple[int, int]]:
                pts = seed_data.get(key, [])
                out = []
                for p in pts:
                    if len(p) >= 2:
                        px = int(p[0] * w) if p[0] <= 1.0 else int(p[0])
                        py = int(p[1] * h) if p[1] <= 1.0 else int(p[1])
                        out.append((px, py))
                return out

            bra_seeds = parse_pts("bra")
            brief_seeds = parse_pts("brief")

            m_bra = (
                cls._grow_garment_region(lab, bgr, bra_seeds, color_tolerance, lace_bridge_size, skin_mask)
                if bra_seeds
                else np.zeros((h, w), dtype=np.uint8)
            )
            m_brief = (
                cls._grow_garment_region(lab, bgr, brief_seeds, color_tolerance, lace_bridge_size, skin_mask)
                if brief_seeds
                else np.zeros((h, w), dtype=np.uint8)
            )

            # Feathering & Boundary Expansion
            m_bra = cls._feather_mask(m_bra, mask_expand, mask_blur)
            m_brief = cls._feather_mask(m_brief, mask_expand, mask_blur)

            # Visualization map
            debug_bgr = bgr.copy()
            debug_bgr[m_bra > 0] = cv2.addWeighted(
                debug_bgr[m_bra > 0], 0.5, np.full_like(debug_bgr[m_bra > 0], (255, 200, 0)), 0.5, 0.0
            )
            debug_bgr[m_brief > 0] = cv2.addWeighted(
                debug_bgr[m_brief > 0], 0.5, np.full_like(debug_bgr[m_brief > 0], (0, 255, 255)), 0.5, 0.0
            )

            for sx, sy in bra_seeds:
                cv2.circle(debug_bgr, (sx, sy), 6, (255, 0, 0), -1)
            for sx, sy in brief_seeds:
                cv2.circle(debug_bgr, (sx, sy), 6, (0, 0, 255), -1)

            debug_rgb = cv2.cvtColor(debug_bgr, cv2.COLOR_BGR2RGB)

            bra_masks.append(torch.from_numpy(m_bra.astype(np.float32) / 255.0).to(device=device, dtype=dtype))
            brief_masks.append(torch.from_numpy(m_brief.astype(np.float32) / 255.0).to(device=device, dtype=dtype))
            debug_maps.append(torch.from_numpy(debug_rgb.astype(np.float32) / 255.0).to(device=device, dtype=dtype))

        all_seed_points = seed_data.get("brief", []) + seed_data.get("bra", [])

        return {
            "ui": {
                "segfit_pro": [{
                    "seed_points": all_seed_points
                }]
            },
            "result": (
                torch.stack(bra_masks, dim=0),
                torch.stack(brief_masks, dim=0),
                torch.stack(debug_maps, dim=0),
            ),
        }