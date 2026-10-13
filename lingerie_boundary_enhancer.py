# ==============================================================================
# File Name:   lingerie_boundary_enhancer.py
# Description: Enhances micro-shadows, seam edges, and texture discrepancies of
#              pale or skin-toned undergarments to allow SAM and SegFormer to
#              reliably segment low-contrast lingerie.
# Proposer:    MeguPaPa3
# Author:      Google Gemini
# Created:     2026-09-17
#
# Change Log:
# Date       Author         Description
# ---------- -------------- ----------------------------------------------------
# 2026-09-17 Google Gemini  Initial release: Added CLAHE contrast enhancement,
#                           shadow ridge extraction, and edge-boost compositing.
# ==============================================================================

from __future__ import annotations

import cv2
import numpy as np
import torch


# ------------------------------------------------------------------------------
# Preprocesses person images to expose low-contrast underwear boundaries.
# Generates high-contrast edge/shadow guidance images so SAM or SegFormer
# can detect nude-tone, beige, or white undergarments without manual editing.
# ------------------------------------------------------------------------------
class LingerieBoundaryEnhancer:
    CATEGORY = "SegFitTools/Image"
    FUNCTION = "enhance_boundary"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("enhanced_image", "shadow_edge_map")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "clahe_clip_limit": (
                    "FLOAT",
                    {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.5},
                ),
                "shadow_boost": (
                    "FLOAT",
                    {"default": 1.5, "min": 0.0, "max": 5.0, "step": 0.1},
                ),
                "edge_sharpness": (
                    "FLOAT",
                    {"default": 1.2, "min": 0.0, "max": 3.0, "step": 0.1},
                ),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, image=None, **_kwargs):
        if image is not None and (
            not isinstance(image, torch.Tensor)
            or image.ndim != 4
            or image.shape[-1] < 3
        ):
            return "[Lingerie Boundary Enhancer] image must be an IMAGE tensor (B,H,W,C)."
        return True

    @staticmethod
    def enhance_boundary(image, clahe_clip_limit, shadow_boost, edge_sharpness):
        if image.ndim != 4 or image.shape[-1] < 3:
            raise ValueError("[Lingerie Boundary Enhancer] Invalid IMAGE tensor shape.")

        batch_size = image.shape[0]
        out_enhanced = []
        out_edges = []

        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip_limit), tileGridSize=(8, 8))

        for b in range(batch_size):
            rgb_tensor = image[b, ..., :3]
            rgb_np = (rgb_tensor.detach().cpu().numpy() * 255.0).round().astype(np.uint8)

            # 1. LAB色空間変換による微小色相・明度差のあぶり出し
            lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)

            # 明度チャンネルにCLAHEを適用（肌と布地の微細な反射の違いを強調）
            l_enhanced = clahe.apply(l_channel)

            # 色度チャンネル（A/B）のダイナミックレンジも拡張
            a_enhanced = clahe.apply(a_channel)
            b_enhanced = clahe.apply(b_channel)

            lab_boosted = cv2.merge([l_enhanced, a_enhanced, b_enhanced])
            enhanced_bgr = cv2.cvtColor(lab_boosted, cv2.COLOR_LAB2RGB)

            # 2. 段差の影（下着の縁・ゴムの食い込み）の抽出
            # モーフォロジー勾配（膨張 - 収縮）で極細のエッジ差分を取得
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph_gradient = cv2.morphologyEx(l_channel, cv2.MORPH_GRADIENT, kernel)

            # 暗いエッジ（影）を検出するブラックハット変換（閉処理 - 元画像）
            blackhat = cv2.morphologyEx(l_channel, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            
            # 影とエッジの合成マップ
            combined_shadow = cv2.addWeighted(morph_gradient, 0.6, blackhat, 0.4 * shadow_boost, 0)
            combined_shadow = np.clip(combined_shadow, 0, 255).astype(np.uint8)

            # 3. 元画像へのエッジ強調合成
            shadow_colored = cv2.cvtColor(combined_shadow, cv2.COLOR_GRAY2RGB)
            if edge_sharpness > 0:
                final_enhanced = cv2.addWeighted(enhanced_bgr, 1.0, shadow_colored, edge_sharpness * 0.5, 0)
            else:
                final_enhanced = enhanced_bgr

            # テンソル変換
            enh_tensor = torch.from_numpy(final_enhanced.astype(np.float32) / 255.0).to(
                device=image.device, dtype=image.dtype
            )
            edge_map_tensor = torch.from_numpy(shadow_colored.astype(np.float32) / 255.0).to(
                device=image.device, dtype=image.dtype
            )

            if image.shape[-1] > 3:
                enh_tensor = torch.cat([enh_tensor, image[b, ..., 3:]], dim=-1)
                edge_map_tensor = torch.cat([edge_map_tensor, image[b, ..., 3:]], dim=-1)

            out_enhanced.append(enh_tensor)
            out_edges.append(edge_map_tensor)

        return (torch.stack(out_enhanced, dim=0), torch.stack(out_edges, dim=0))