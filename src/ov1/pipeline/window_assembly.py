"""Orchestration glue: a `RingBuffer` window of `SyncedFramePair` to an
assembled spec 4.2 inference tensor (bead ocean-vision-v1-0h8).

`ocean-vision-v1-3fy` (`ov1.tensor.assemble`) only stacks already-computed
per-frame `FrameChannels`; nothing yet turns a window of raw
`SyncedFramePair` into that list. This module is that glue: each pair's RGB
frame goes through `LabNormalizer`, its NIR frame's four polarization planes
go through `compute_stokes`/`compute_dolp`, `SpecularGlareMasker` suppresses
glare on the L plane, and `FrameChannels.from_lab_and_dolp()` packages the
result -- one call per frame, `InferenceTensorAssembler.assemble()` for the
window.

Two things this module decides that nothing upstream had fixed yet:

1. Where `LabNormalizer`/`SpecularGlareMasker`'s per-site calibration
   (`clip_limit`, `tile_grid_size`, `dolp_threshold`) is loaded from:
   `SiteCalibrationConfig.from_json_file()`, a flat JSON file, following the
   plain-`json`-file convention `ov1.data.manifest.load_manifest` already
   uses for provenance records. There is still no built-in default -- a
   missing config file or a missing key raises, exactly as constructing
   `LabNormalizer`/`SpecularGlareMasker` directly without those knobs would.
2. How a NIR `Frame`'s `data` array carries the four 0/45/90/135-degree
   polarization intensity planes `compute_stokes` needs.
   `ov1.preprocess.dolp`'s own docstring says going from the 850nm sensor's
   raw mosaic to four demosaiced planes is "a hardware-driver concern this
   repo has not built yet" and that module "starts from the four planes,
   however they arrive." No real driver exists in this repo to define how
   they arrive, so `split_nir_polarization_planes()` below fixes a
   convention -- NIR `Frame.data` is an `(H, W, 4)` array, planes stacked on
   the last axis in `[i0, i45, i90, i135]` order -- documented here as a
   convention this bead is choosing, not a fact read off real hardware.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ov1.ingress.sync import SyncedFramePair
from ov1.preprocess.dolp import SpecularGlareMasker, compute_dolp, compute_stokes
from ov1.preprocess.lab import LabNormalizer
from ov1.tensor.assemble import AssembledTensor, FrameChannels, InferenceTensorAssembler

#: NIR `Frame.data`'s last-axis order this module requires. A convention
#: fixed by this bead (module docstring point 2), not a hardware fact --
#: there is no real 850nm capture backend in this repo yet to read the order
#: off of.
NIR_PLANE_ORDER = ("i0", "i45", "i90", "i135")

_SITE_CONFIG_REQUIRED_FIELDS = ("clip_limit", "tile_grid_size", "dolp_threshold")


class SiteConfigError(ValueError):
    """Raised when a site calibration config file is missing, malformed, or
    missing a required field. Per this repo's fail-loud rule, a broken
    config must stop the pipeline from starting, not fall back to a guessed
    `clip_limit`/`tile_grid_size`/`dolp_threshold`."""


@dataclass(frozen=True)
class SiteCalibrationConfig:
    """The per-installation calibration knobs `LabNormalizer` and
    `SpecularGlareMasker` require (their own docstrings: "not algorithm
    constants," "no built-in default"). Values come from a site's own
    baseline-footage calibration, never from this module.
    """

    clip_limit: float
    tile_grid_size: tuple[int, int]
    dolp_threshold: float

    @classmethod
    def from_mapping(cls, data: dict) -> "SiteCalibrationConfig":
        missing = [f for f in _SITE_CONFIG_REQUIRED_FIELDS if f not in data]
        if missing:
            raise SiteConfigError(
                f"site calibration config missing required field(s): {', '.join(missing)}"
            )
        tile_grid_size = data["tile_grid_size"]
        if (
            not isinstance(tile_grid_size, (list, tuple))
            or len(tile_grid_size) != 2
        ):
            raise SiteConfigError(
                f"tile_grid_size must be a 2-element array, got {tile_grid_size!r}"
            )
        return cls(
            clip_limit=float(data["clip_limit"]),
            tile_grid_size=(int(tile_grid_size[0]), int(tile_grid_size[1])),
            dolp_threshold=float(data["dolp_threshold"]),
        )

    @classmethod
    def from_json_file(cls, path: Path | str) -> "SiteCalibrationConfig":
        """Load a site calibration config from a flat JSON file, e.g.::

            {"clip_limit": 2.0, "tile_grid_size": [8, 8], "dolp_threshold": 0.5}

        Raises `SiteConfigError` if the file is missing, is not valid JSON,
        or omits a required field.
        """
        path = Path(path)
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SiteConfigError(f"could not read site config {path}: {exc}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SiteConfigError(f"{path}: invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise SiteConfigError(f"{path}: expected a JSON object, got {type(data).__name__}")
        return cls.from_mapping(data)

    def build_lab_normalizer(self) -> LabNormalizer:
        return LabNormalizer(clip_limit=self.clip_limit, tile_grid_size=self.tile_grid_size)

    def build_glare_masker(self) -> SpecularGlareMasker:
        return SpecularGlareMasker(dolp_threshold=self.dolp_threshold)


def split_nir_polarization_planes(
    nir_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a NIR `Frame.data` array into its four polarization intensity
    planes, per `NIR_PLANE_ORDER`.

    Raises `ValueError` if `nir_data` is not `(H, W, 4)` -- never reshapes or
    pads a mismatched array into agreement.
    """
    if nir_data.ndim != 3 or nir_data.shape[2] != 4:
        raise ValueError(
            f"expected NIR frame data shaped (H, W, 4) for {NIR_PLANE_ORDER}, "
            f"got shape {nir_data.shape}"
        )
    return (
        nir_data[..., 0],
        nir_data[..., 1],
        nir_data[..., 2],
        nir_data[..., 3],
    )


class FrameChannelsPipeline:
    """Converts one `SyncedFramePair` (or a window of them) into the
    `FrameChannels` `InferenceTensorAssembler` consumes, using a fixed pair
    of calibrated `LabNormalizer`/`SpecularGlareMasker` instances.
    """

    def __init__(self, lab_normalizer: LabNormalizer, glare_masker: SpecularGlareMasker):
        self._lab_normalizer = lab_normalizer
        self._glare_masker = glare_masker

    @classmethod
    def from_site_config(cls, config: SiteCalibrationConfig) -> "FrameChannelsPipeline":
        return cls(config.build_lab_normalizer(), config.build_glare_masker())

    def convert(self, pair: SyncedFramePair) -> FrameChannels:
        """Run one synchronized frame pair's RGB frame through
        `LabNormalizer` and NIR frame through `compute_stokes`/
        `compute_dolp`, then package the result via
        `FrameChannels.from_lab_and_dolp()` with glare suppression applied.

        Propagates whatever `LabNormalizer.normalize`, `compute_stokes`,
        `compute_dolp`, or `split_nir_polarization_planes` raise on bad
        input -- this method adds no fallback of its own.
        """
        lab = self._lab_normalizer.normalize(pair.rgb.data)
        i0, i45, i90, i135 = split_nir_polarization_planes(pair.nir.data)
        stokes = compute_stokes(i0, i45, i90, i135)
        dolp = compute_dolp(stokes)
        return FrameChannels.from_lab_and_dolp(lab, dolp, glare_masker=self._glare_masker)

    def convert_window(self, pairs: Sequence[SyncedFramePair]) -> list[FrameChannels]:
        """`convert()` applied to each pair in the window, oldest first."""
        return [self.convert(pair) for pair in pairs]


def assemble_tensor_from_window(
    pairs: Sequence[SyncedFramePair],
    pipeline: FrameChannelsPipeline,
    assembler: InferenceTensorAssembler,
    *,
    pin: bool = True,
) -> AssembledTensor:
    """End-to-end: a `RingBuffer.window()` snapshot straight through to an
    assembled spec 4.2 inference tensor.

    Raises whatever `FrameChannelsPipeline.convert` or
    `InferenceTensorAssembler.assemble` raise -- notably
    `ov1.tensor.assemble.TensorAssemblyError` if `pairs` is not exactly
    `assembler`'s configured `window_frames` long, which is the normal
    outcome of calling this before the ring buffer is full.
    """
    frame_channels = pipeline.convert_window(pairs)
    return assembler.assemble(frame_channels, pin=pin)
