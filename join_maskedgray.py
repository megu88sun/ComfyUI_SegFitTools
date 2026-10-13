# ==============================================================================
# File Name:    join_maskedgray.py
# Description:  Composite the source image by using a grayscale version of its mask image.
# Proposer:     MeguPaPa3
# Author:       MeguPaPa3
# Created:      2026-07-24
#
# Change Log:
# Date        Author        Description
# ----------  ------------  --------------------------------------------------
# 2026-07-24  MeguPaPa3     Initial release.
# 2026-07-24  OpenAI Codex  Fixed ComfyUI schema and PyTorch tensor processing.
# ==============================================================================

# import cv2
import torch
import torch.nn.functional as F
# import numpy as np

# ------------------------------------------------------------------------------
# PyTorch is used to check the image and mask, while NumPy is used for
#  compositing and converting them to black-and-white (luminance).
# ------------------------------------------------------------------------------
class JoinMaskedAreaGray:
    CATEGORY = "SegFitTools/Image"
    FUNCTION = "maskedgray"
    # RETURN_TYPE = ("IMAGE")
    # RETURN_NAME = ("image")
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        # return {
        #     "require": {
        #         "image": {"IMAGE",},
        #         "mask": {"MASK"},
        #     }
        # }
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, image=None, mask=None):
        if image is not None and (
            not isinstance(image, torch.Tensor)
            or image.ndim != 4
            or image.shape[-1] < 3
        ):
            # return "[Join Masked Area Gray] image must be an INPUT tensor (B,H,W,C)."
            return "[Join Masked Area Gray] image must be an IMAGE tensor (B,H,W,C)."
        if mask is not None and (
            not isinstance(mask, torch.Tensor) or mask.ndim not in (2, 3)
        ):
            return "[Join Masked Area Gray] mask must be a MASK tensor (B,H,W)."
        return True

    @staticmethod
    def maskedgray(image, mask):
        if image.ndim != 4 or image.shape[-1] < 3:
            raise ValueError("[Join Masked Area Gray] Invalid IMAGE tensor shape.")
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim != 3:
            raise ValueError("[Join Masked Area Gray] Invalid MASK tensor shape.")
        if mask.shape[0] not in (1, image.shape[0]):
            raise ValueError(
                "[Join Masked Area Gray] MASK batch must be one or match the IMAGE batch."
            )
        if not torch.isfinite(image).all() or not torch.isfinite(mask).all():
            raise ValueError("[Join Masked Area Gray] IMAGE and MASK must contain finite values.")

        # Previous NumPy/OpenCV implementation retained temporarily for review.
        #
        # # 1. Loading the source image and mask (internally handled as tensors in ComfyUI, but converted to NumPy here)
        # # image: (H, W, 3) / mask: (H, W) -> Normalized to the range of 0 to 1
        # mask_normalized = mask.astype(np.float32) / 255.0
        # mask_3d = np.expand_dims(mask_normalized, axis=2)  # (H, W, 1)
        #
        # # 2. Create a black-and-white (luminance) image containing only the underwear area.
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # 3 Channel
        #
        # # 3. Set a base gray color that blends with the surrounding skin (e.g., a mid-tone brightness
        # #    of 128, or the average value of the surrounding skin).
        # # To utilize the shading, blend with a black-and-white image (e.g., 50% each) to neutralize
        # # the colors.
        # target_shading = cv2.addWeighted(gray_3ch, 0.5, np.full_like(image, 128), 0.5, 0)
        #
        # # 4. Use a mask to alpha-blend *only the underwear area* onto the original, intact image.
        # output_image = (image * (1.0 - mask_3d) + target_shading * mask_3d).astype(np.uint8)
        # return (output_image,)

        # Normalize the ComfyUI MASK without changing its existing 0.0 to 1.0 range.
        mask_value = mask.to(device=image.device, dtype=image.dtype).clamp(0.0, 1.0)
        if mask_value.shape[1:3] != image.shape[1:3]:
            mask_value = F.interpolate(
                mask_value.unsqueeze(1),
                size=image.shape[1:3],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        if mask_value.shape[0] == 1 and image.shape[0] > 1:
            mask_value = mask_value.expand(image.shape[0], -1, -1)

        # Calculate RGB luminance and blend it equally with middle gray.
        rgb = image[..., :3]
        luminance_weights = torch.tensor(
            [0.299, 0.587, 0.114],
            device=image.device,
            dtype=image.dtype,
        ).view(1, 1, 1, 3)
        luminance = (rgb * luminance_weights).sum(dim=-1, keepdim=True)
        middle_gray = torch.tensor(
            128.0 / 255.0,
            device=image.device,
            dtype=image.dtype,
        )
        target_shading = luminance * 0.5 + middle_gray * 0.5

        # Replace only the masked RGB area and preserve any additional channels.
        output_image = image.clone()
        alpha = mask_value.unsqueeze(-1)
        output_image[..., :3] = (
            rgb * (1.0 - alpha)
            + target_shading.expand_as(rgb) * alpha
        )
        return (output_image,)
