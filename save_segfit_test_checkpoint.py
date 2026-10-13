# ==============================================================================
# File Name:    save_segfit_test_checkpoint.py
# Description:  Saves a SegFit TEST checkpoint.
#               Full analysis data is stored with torch.save().
#               Preview image and readable JSON information are also saved.
#               The input data and preview are passed through unchanged.
# Proposer:     MeguPaPa3
# Author:       OpenAI ChatGPT
# Created:      2026-09-27
# ==============================================================================

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import numpy as np
import torch
from PIL import Image

try:
    import folder_paths
except ImportError:
    # Standalone/pytest fallback when running outside the full ComfyUI runtime
    from unittest.mock import MagicMock

    folder_paths = MagicMock()

LOGGER = logging.getLogger(
    "SegFitTools.SaveSegFitTestCheckpoint"
)


# ==============================================================================
# ComfyUI wildcard type
# ==============================================================================

class AnyType(str):

    def __ne__(
        self,
        __value: object,
    ) -> bool:

        return False


ANY_TYPE = AnyType("*")


# ==============================================================================
# Node
# ==============================================================================

class SaveSegFitTestCheckpoint:

    CATEGORY = "SegFitTools/TEST"
    FUNCTION = "save_checkpoint"

    RETURN_TYPES = (
        "IMAGE",
        ANY_TYPE,
    )

    RETURN_NAMES = (
        "preview",
        "data",
    )

    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "preview": (
                    "IMAGE",
                ),
                "data": (
                    ANY_TYPE,
                ),
                "checkpoint_name": (
                    "STRING",
                    {
                        "default":
                            "SegFit_TEST_Checkpoint",
                        "multiline":
                            False,
                    },
                ),
            }
        }

    # --------------------------------------------------------------------------
    # TEST directory
    # --------------------------------------------------------------------------

    @staticmethod
    def _get_test_directory() -> str:

        return os.path.join(
            folder_paths.get_input_directory(),
            "SegFit_TEST",
        )

    # --------------------------------------------------------------------------
    # Safe filename
    # --------------------------------------------------------------------------

    @staticmethod
    def _safe_name(
        name: str,
    ) -> str:

        name = str(
            name
        ).strip()

        if not name:

            name = (
                "SegFit_TEST_Checkpoint"
            )

        #
        # Invalid Windows filename characters.
        #

        name = re.sub(
            r'[<>:"/\\|?*]',
            "_",
            name,
        )

        #
        # Windows does not like trailing spaces / dots.
        #

        name = name.rstrip(
            " ."
        )

        if not name:

            name = (
                "SegFit_TEST_Checkpoint"
            )

        return name

    # --------------------------------------------------------------------------
    # Preview conversion
    # --------------------------------------------------------------------------

    @staticmethod
    def _preview_to_image(
        preview,
    ) -> Image.Image:

        if not isinstance(
            preview,
            torch.Tensor,
        ):

            raise TypeError(
                "preview must be torch.Tensor"
            )

        image = (
            preview
            .detach()
            .cpu()
        )

        #
        # ComfyUI IMAGE:
        #
        # [B,H,W,C]
        #

        if image.ndim != 4:

            raise ValueError(
                "preview must be ComfyUI IMAGE "
                f"[B,H,W,C], got {tuple(image.shape)}"
            )

        if image.shape[0] < 1:

            raise ValueError(
                "preview contains no image"
            )

        image = image[0]

        if image.shape[-1] not in (
            1,
            3,
            4,
        ):

            raise ValueError(
                "preview channel count must be "
                "1, 3, or 4"
            )

        array = (
            image
            .clamp(
                0.0,
                1.0,
            )
            .numpy()
        )

        array = (
            array * 255.0
        ).round().astype(
            np.uint8
        )

        if array.shape[-1] == 1:

            array = array[
                :,
                :,
                0
            ]

            return Image.fromarray(
                array,
                mode="L",
            )

        if array.shape[-1] == 4:

            return Image.fromarray(
                array,
                mode="RGBA",
            )

        return Image.fromarray(
            array,
            mode="RGB",
        )

    # --------------------------------------------------------------------------
    # JSON-safe diagnostic conversion
    # --------------------------------------------------------------------------

    @classmethod
    def _json_safe(
        cls,
        value: Any,
    ):

        #
        # IMPORTANT:
        #
        # JSON is NOT the checkpoint data.
        #
        # The .pt file is the real checkpoint.
        # JSON exists only so a human can inspect
        # the stored analysis information.
        #

        if isinstance(
            value,
            torch.Tensor,
        ):

            tensor = (
                value
                .detach()
                .cpu()
            )

            info = {
                "__type__":
                    "torch.Tensor",

                "shape":
                    list(
                        tensor.shape
                    ),

                "dtype":
                    str(
                        tensor.dtype
                    ),

                "numel":
                    int(
                        tensor.numel()
                    ),
            }

            #
            # Small tensors are useful to inspect.
            # Large tensors are deliberately NOT
            # expanded into JSON.
            #

            if tensor.numel() <= 256:

                info[
                    "data"
                ] = tensor.tolist()

            else:

                info[
                    "data"
                ] = (
                    "<stored completely in .pt>"
                )

            return info

        if isinstance(
            value,
            np.ndarray,
        ):

            info = {
                "__type__":
                    "numpy.ndarray",

                "shape":
                    list(
                        value.shape
                    ),

                "dtype":
                    str(
                        value.dtype
                    ),

                "size":
                    int(
                        value.size
                    ),
            }

            if value.size <= 256:

                info[
                    "data"
                ] = value.tolist()

            else:

                info[
                    "data"
                ] = (
                    "<stored completely in .pt>"
                )

            return info

        if isinstance(
            value,
            dict,
        ):

            return {
                str(key):
                    cls._json_safe(
                        item
                    )
                for key, item
                in value.items()
            }

        if isinstance(
            value,
            list,
        ):

            return [
                cls._json_safe(
                    item
                )
                for item
                in value
            ]

        if isinstance(
            value,
            tuple,
        ):

            return {
                "__type__":
                    "tuple",

                "data": [
                    cls._json_safe(
                        item
                    )
                    for item
                    in value
                ],
            }

        if isinstance(
            value,
            set,
        ):

            return {
                "__type__":
                    "set",

                "data": [
                    cls._json_safe(
                        item
                    )
                    for item
                    in value
                ],
            }

        if isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
            ),
        ):

            return value

        if value is None:

            return None

        #
        # Diagnostic fallback only.
        #

        return {
            "__type__":
                type(
                    value
                ).__name__,

            "repr":
                repr(
                    value
                ),
        }

    # --------------------------------------------------------------------------
    # Main
    # --------------------------------------------------------------------------

    def save_checkpoint(
        self,
        preview,
        data,
        checkpoint_name,
    ):

        stage = "START"

        try:

            # ==============================================================
            # Directory
            # ==============================================================

            stage = "DIRECTORY"

            directory = (
                self._get_test_directory()
            )

            os.makedirs(
                directory,
                exist_ok=True,
            )

            safe_name = (
                self._safe_name(
                    checkpoint_name
                )
            )

            pt_path = os.path.join(
                directory,
                safe_name + ".pt",
            )

            png_path = os.path.join(
                directory,
                safe_name + ".png",
            )

            json_path = os.path.join(
                directory,
                safe_name + ".json",
            )

            LOGGER.info(
                "[SegFitTools]"
                "[SaveSegFitTestCheckpoint] "
                "start name=%s",
                safe_name,
            )

            # ==============================================================
            # Save complete data
            # ==============================================================

            stage = "SAVE_DATA"

            #
            # This is the real checkpoint.
            #
            # Tensor data is NOT converted to text,
            # NOT truncated, and NOT passed through
            # ComfyUI display representation.
            #

            torch.save(
                data,
                pt_path,
            )

            # ==============================================================
            # Save preview
            # ==============================================================

            stage = "SAVE_PREVIEW"

            image = (
                self._preview_to_image(
                    preview
                )
            )

            image.save(
                png_path,
                format="PNG",
            )

            # ==============================================================
            # Save readable JSON
            # ==============================================================

            stage = "SAVE_JSON"

            json_data = {
                "checkpoint_name":
                    safe_name,

                "checkpoint_file":
                    os.path.basename(
                        pt_path
                    ),

                "preview_file":
                    os.path.basename(
                        png_path
                    ),

                "data":
                    self._json_safe(
                        data
                    ),
            }

            with open(
                json_path,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    json_data,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            # ==============================================================
            # Complete
            # ==============================================================

            LOGGER.info(
                "[SegFitTools]"
                "[SaveSegFitTestCheckpoint] "
                "completed name=%s",
                safe_name,
            )

            #
            # Pass-through.
            #
            # Nothing is modified.
            #

            return (
                preview,
                data,
            )

        except Exception as error:

            #
            # SegFit side stays short.
            # ComfyUI itself will print its traceback.
            #

            LOGGER.error(
                "[SegFitTools]"
                "[SaveSegFitTestCheckpoint]"
                "[ERROR] "
                "stage=%s reason=%s",
                stage,
                error,
            )

            raise