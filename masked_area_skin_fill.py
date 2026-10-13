# ==============================================================================
# File Name: masked_area_skin_fill.py
# Description: Synthesizes a seamless skin-toned underlay across masked garment
#              regions using classical fluid/harmonic inpainting (Telea/Navier-Stokes)
#              and low-frequency smoothing. Eliminates latent contamination and
#              ghosting of prior undergarments without generating explicit base bodies.
# Proposer: MeguPaPa3
# Author: Google Gemini
# Created: 2026-09-17
#
# Change Log:
# Date       Author         Description
# ---------- -------------- ----------------------------------------------------
# 2026-09-17 Google Gemini  Initial compliant implementation adhering to
#                           SegFitTools coding guidelines and strict type guards.
# ==============================================================================

from __future__ import annotations

import cv2
import numpy as np
import torch


# ------------------------------------------------------------------------------
# Inpaints masked undergarment regions using boundary skin-tone diffusion.
# Serves as an untextured, non-explicit canvas to prevent diffusion model
# latent memory retrieval of previous clothing articles.
# ------------------------------------------------------------------------------
class MaskedAreaSkinFill:
    CATEGORY = "SegFitTools/Preprocess"
    FUNCTION = "apply_skin_fill"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("underlay_image", "dilated_mask")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "expand_pixels": (
                    "INT",
                    {
                        "default": 6,
                        "min": 0,
                        "max": 64,
                        "step": 1,
                    },
                ),
                "inpaint_method": (
                    ["telea", "navier_stokes"],
                    {"default": "telea"},
                ),
                "inpaint_radius": (
                    "INT",
                    {
                        "default": 7,
                        "min": 1,
                        "max": 50,
                        "step": 1,
                    },
                ),
                "smooth_strength": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                    },
                ),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        image=None,
        mask=None,
        expand_pixels=None,
        inpaint_method=None,
        inpaint_radius=None,
        smooth_strength=None,
        **_kwargs,
    ):
        if image is not None:
            if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[-1] < 3:
                return "[MaskedAreaSkinFill] 'image' must be an IMAGE tensor (B, H, W, C)."
        if mask is not None:
            if not isinstance(mask, torch.Tensor) or mask.ndim not in (2, 3):
                return "[MaskedAreaSkinFill] 'mask' must be a MASK tensor (B, H, W) or (H, W)."
        if expand_pixels is not None and (expand_pixels < 0 or expand_pixels > 64):
            return "[MaskedAreaSkinFill] 'expand_pixels' must be between 0 and 64."
        if inpaint_method is not None and inpaint_method not in ("telea", "navier_stokes"):
            return "[MaskedAreaSkinFill] 'inpaint_method' must be 'telea' or 'navier_stokes'."
        if inpaint_radius is not None and (inpaint_radius < 1 or inpaint_radius > 50):
            return "[MaskedAreaSkinFill] 'inpaint_radius' must be between 1 and 50."
        if smooth_strength is not None and (smooth_strength < 0.0 or smooth_strength > 1.0):
            return "[MaskedAreaSkinFill] 'smooth_strength' must be between 0.0 and 1.0."
        return True

    @staticmethod
    def apply_skin_fill(
        image: torch.Tensor,
        mask: torch.Tensor,
        expand_pixels: int,
        inpaint_method: str,
        inpaint_radius: int,
        smooth_strength: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 4 or image.shape[-1] < 3:
            raise ValueError("[MaskedAreaSkinFill] Expected image shape (B, H, W, C).")

        batch_size = image.shape[0]
        device = image.device
        dtype = image.dtype

        cv_flag = cv2.INPAINT_TELEA if inpaint_method == "telea" else cv2.INPAINT_NS

        # Align mask to (B, H, W)
        if mask.ndim == 2:
            mask_batch = mask.unsqueeze(0).repeat(batch_size, 1, 1)
        elif mask.ndim == 3 and mask.shape[0] == 1 and batch_size > 1:
            mask_batch = mask.repeat(batch_size, 1, 1)
        elif mask.ndim == 3 and mask.shape[0] == batch_size:
            mask_batch = mask
        else:
            raise ValueError(
                f"[MaskedAreaSkinFill] Mask batch dimension ({mask.shape[0]}) does not match image batch ({batch_size})."
            )

        out_images: list[torch.Tensor] = []
        out_masks: list[torch.Tensor] = []

        for b in range(batch_size):
            rgb_slice = image[b, ..., :3]
            img_np = (rgb_slice.detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)

            m_slice = mask_batch[b]
            m_np = (m_slice.detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)

            # Resize mask if spatial dimensions mismatch
            if m_np.shape[:2] != img_np.shape[:2]:
                m_np = cv2.resize(
                    m_np,
                    (img_np.shape[1], img_np.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            # Expand mask boundary to swallow seam edges
            if expand_pixels > 0:
                ksize = expand_pixels * 2 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
                dilated_m_np = cv2.dilate(m_np, kernel, iterations=1)
            else:
                dilated_m_np = m_np.copy()

            # Pass through if mask is empty
            if cv2.countNonZero(dilated_m_np) == 0:
                out_images.append(image[b])
                mask_t = torch.from_numpy(dilated_m_np.astype(np.float32) / 255.0).to(
                    device=device, dtype=dtype
                )
                out_masks.append(mask_t)
                continue

            # Classical Navier-Stokes / Telea inpainting across skin boundary
            bgr_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            inpainted_bgr = cv2.inpaint(
                bgr_np,
                dilated_m_np,
                inpaintRadius=int(inpaint_radius),
                flags=cv_flag,
            )
            inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)

            # Low-frequency smoothing to flatten residual garment seam shadows
            if smooth_strength > 0.0:
                calc_k = int(max(15, (img_np.shape[0] * 0.02) // 2 * 2 + 1))
                blurred_rgb = cv2.GaussianBlur(inpainted_rgb, (calc_k, calc_k), 0)
                alpha = float(smooth_strength)
                inpainted_rgb = cv2.addWeighted(blurred_rgb, alpha, inpainted_rgb, 1.0 - alpha, 0.0)

            # Soft feathering at boundary
            mask_float = dilated_m_np.astype(np.float32) / 255.0
            feathered_mask = cv2.GaussianBlur(mask_float, (5, 5), 0)[..., None]

            composed = (
                inpainted_rgb.astype(np.float32) * feathered_mask
                + img_np.astype(np.float32) * (1.0 - feathered_mask)
            ).clip(0, 255).astype(np.uint8)

            composed_tensor = torch.from_numpy(composed.astype(np.float32) / 255.0).to(
                device=device, dtype=dtype
            )

            # Preserve alpha channels if original image had them
            if image.shape[-1] > 3:
                composed_tensor = torch.cat([composed_tensor, image[b, ..., 3:]], dim=-1)

            dilated_mask_tensor = torch.from_numpy(dilated_m_np.astype(np.float32) / 255.0).to(
                device=device, dtype=dtype
            )

            out_images.append(composed_tensor)
            out_masks.append(dilated_mask_tensor)

        return (torch.stack(out_images, dim=0), torch.stack(out_masks, dim=0))