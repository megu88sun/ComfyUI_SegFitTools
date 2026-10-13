# ==============================================================================
# File Name:    load_segfit_test_checkpoint.py
# Description:  Loads a SegFit TEST checkpoint and its preview image.
#               Intended to restart TEST analysis from an intermediate stage
#               without running all previous analysis nodes again.
# Proposer:     MeguPaPa3
# Author:       OpenAI ChatGPT
# Created:      2026-09-27
# ==============================================================================

from __future__ import annotations

import logging
import os

import numpy as np
import torch
from PIL import Image

import folder_paths


LOGGER = logging.getLogger(
    "SegFitTools.LoadSegFitTestCheckpoint"
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

class LoadSegFitTestCheckpoint:

    CATEGORY = "SegFitTools/TEST"
    FUNCTION = "load_checkpoint"

    RETURN_TYPES = (
        "IMAGE",
        ANY_TYPE,
    )

    RETURN_NAMES = (
        "preview",
        "data",
    )

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
    # Checkpoint list
    # --------------------------------------------------------------------------

    @classmethod
    def _get_checkpoint_files(
        cls,
    ):

        directory = (
            cls._get_test_directory()
        )

        if not os.path.isdir(
            directory
        ):

            return []

        files = []

        for name in os.listdir(
            directory
        ):

            full_path = os.path.join(
                directory,
                name,
            )

            if not os.path.isfile(
                full_path
            ):
                continue

            if not name.lower().endswith(
                ".pt"
            ):
                continue

            files.append(
                name
            )

        files.sort(
            key=str.lower
        )

        return files

    # --------------------------------------------------------------------------
    # Inputs
    # --------------------------------------------------------------------------

    @classmethod
    def INPUT_TYPES(
        cls,
    ):

        files = (
            cls._get_checkpoint_files()
        )

        if not files:

            files = [
                "<NO CHECKPOINT>"
            ]

        return {
            "required": {
                "checkpoint_file": (
                    files,
                ),
            }
        }

    # --------------------------------------------------------------------------
    # Reload when checkpoint changes
    # --------------------------------------------------------------------------

    @classmethod
    def IS_CHANGED(
        cls,
        checkpoint_file,
    ):

        if (
            not checkpoint_file
            or checkpoint_file
            == "<NO CHECKPOINT>"
        ):

            return float(
                "nan"
            )

        directory = (
            cls._get_test_directory()
        )

        pt_path = os.path.join(
            directory,
            checkpoint_file,
        )

        try:

            stat = os.stat(
                pt_path
            )

            return (
                f"{checkpoint_file}:"
                f"{stat.st_mtime_ns}:"
                f"{stat.st_size}"
            )

        except OSError:

            return float(
                "nan"
            )

    # --------------------------------------------------------------------------
    # Path safety
    # --------------------------------------------------------------------------

    @classmethod
    def _safe_checkpoint_path(
        cls,
        filename: str,
    ) -> str:

        directory = os.path.abspath(
            cls._get_test_directory()
        )

        path = os.path.abspath(
            os.path.join(
                directory,
                filename,
            )
        )

        try:

            common = os.path.commonpath(
                [
                    directory,
                    path,
                ]
            )

        except ValueError as error:

            raise ValueError(
                "Invalid checkpoint path"
            ) from error

        if common != directory:

            raise ValueError(
                "Checkpoint path escaped "
                "input/SegFit_TEST"
            )

        return path

    # --------------------------------------------------------------------------
    # Load preview
    # --------------------------------------------------------------------------

    @staticmethod
    def _load_preview(
        png_path: str,
    ) -> torch.Tensor:

        if not os.path.isfile(
            png_path
        ):

            #
            # Preview missing should not destroy
            # the actual checkpoint data load.
            #
            # Return a small black IMAGE instead.
            #

            LOGGER.warning(
                "[SegFitTools]"
                "[LoadSegFitTestCheckpoint] "
                "preview missing file=%s",
                os.path.basename(
                    png_path
                ),
            )

            return torch.zeros(
                (
                    1,
                    64,
                    64,
                    3,
                ),
                dtype=torch.float32,
            )

        with Image.open(
            png_path
        ) as image:

            #
            # Preview is diagnostic.
            # RGB is sufficient here.
            #

            image = image.convert(
                "RGB"
            )

            array = np.asarray(
                image
            ).astype(
                np.float32
            )

        array /= 255.0

        return (
            torch.from_numpy(
                array
            )
            .unsqueeze(0)
            .float()
        )

    # --------------------------------------------------------------------------
    # Main
    # --------------------------------------------------------------------------

    def load_checkpoint(
        self,
        checkpoint_file,
    ):

        stage = "START"

        try:

            # ==============================================================
            # Validate
            # ==============================================================

            stage = "VALIDATE"

            if (
                not checkpoint_file
                or checkpoint_file
                == "<NO CHECKPOINT>"
            ):

                raise FileNotFoundError(
                    "No SegFit TEST checkpoint found"
                )

            if not checkpoint_file.lower().endswith(
                ".pt"
            ):

                raise ValueError(
                    "Checkpoint must be a .pt file"
                )

            # ==============================================================
            # Paths
            # ==============================================================

            stage = "PATH"

            pt_path = (
                self._safe_checkpoint_path(
                    checkpoint_file
                )
            )

            if not os.path.isfile(
                pt_path
            ):

                raise FileNotFoundError(
                    "Checkpoint not found: "
                    f"{checkpoint_file}"
                )

            base_name = os.path.splitext(
                checkpoint_file
            )[0]

            png_path = os.path.join(
                self._get_test_directory(),
                base_name + ".png",
            )

            LOGGER.info(
                "[SegFitTools]"
                "[LoadSegFitTestCheckpoint] "
                "start file=%s",
                checkpoint_file,
            )

            # ==============================================================
            # Load complete data
            # ==============================================================

            stage = "LOAD_DATA"

            #
            # This file was created locally by the matching
            # SegFit TEST checkpoint saver.
            #
            # weights_only=False is required because the checkpoint
            # contains ordinary Python dict/list structures in addition
            # to tensors.
            #

            data = torch.load(
                pt_path,
                map_location="cpu",
                weights_only=False,
            )

            # ==============================================================
            # Load preview
            # ==============================================================

            stage = "LOAD_PREVIEW"

            preview = (
                self._load_preview(
                    png_path
                )
            )

            # ==============================================================
            # Complete
            # ==============================================================

            LOGGER.info(
                "[SegFitTools]"
                "[LoadSegFitTestCheckpoint] "
                "completed file=%s type=%s",
                checkpoint_file,
                type(
                    data
                ).__name__,
            )

            return (
                preview,
                data,
            )

        except Exception as error:

            LOGGER.error(
                "[SegFitTools]"
                "[LoadSegFitTestCheckpoint]"
                "[ERROR] "
                "stage=%s file=%s reason=%s",
                stage,
                checkpoint_file,
                error,
            )

            raise