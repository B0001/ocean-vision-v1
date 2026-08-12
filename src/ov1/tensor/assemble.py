"""Assembly of the mandated (B, C, T, H, W) = (1, 4, 120, 1080, 1920)
inference tensor (spec 4.2).

Per-frame [Luminance, a*, b*, DoLP] planes -- `LabNormalizer`'s L/a*/b*
output (ocean-vision-v1-3qj) and `compute_dolp`'s map (ocean-vision-v1-bze)
-- arrive at native ingress resolution. This module downscales each plane to
the spec's 1080p working resolution, stacks the fixed [L, a*, b*, DoLP]
channel order, stacks the 120-frame window on the T axis, and produces a
pinned-memory `torch.Tensor` for the Jetson inference handoff.

One thing this module deliberately does NOT do:
- It does not compute Lab or DoLP itself -- callers pass already-computed
  per-frame `FrameChannels`. Going from raw RGB/NIR sensor frames to those
  planes is `LabNormalizer.normalize()` and `compute_dolp()`'s job.

Channel-scale normalization (bead ocean-vision-v1-3hw): L/a*/b* arrive uint8
in native [0, 255]; DoLP arrives float64 in native [0, 1]. Stacking those
without a common scale hands the Conv3D autoencoder (ocean-vision-v1-133) a
tensor where one channel's raw magnitude is up to 255x the other three --
plausibly harmful to training regardless of what per-channel weighting the
model eventually learns. This module rescales L/a*/b* onto DoLP's [0, 1]
range by dividing by 255.0 (`_UINT8_TO_UNIT_SCALE` below), not by z-scoring
against baseline-footage statistics: a fixed uint8->[0,1] rescaling is a
property of the encoding (identical at every site), where a z-score's
per-channel mean/std would be a per-site calibration knob this container has
no baseline footage to fit -- exactly the kind of constant this repo's
conventions say must come from a site config, not a guess baked in here. If
133's training shows the model needs per-channel standardization instead,
that is a training-input transform to add in front of the autoencoder, not a
reason to move site-specific statistics into this general assembly path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np
import torch

from ov1.preprocess.dolp import SpecularGlareMasker
from ov1.preprocess.lab import LabFrame

#: Spec 4.2's mandated (C, T, H, W) shape components. Architectural
#: constants fixing this pipeline's contract with the inference model
#: (ocean-vision-v1-133) -- not measurements, and not per-site calibration
#: knobs, so unlike `LabNormalizer`'s CLAHE parameters they get a built-in
#: default.
SPEC_CHANNELS = 4
SPEC_TEMPORAL_WINDOW_FRAMES = 120
SPEC_HEIGHT = 1080
SPEC_WIDTH = 1920

#: Fixed channel order, spec 4.2: "[Luminance, a*, b*, DoLP]".
CHANNEL_ORDER = ("L", "a*", "b*", "DoLP")
CHANNEL_L = 0
CHANNEL_A = 1
CHANNEL_B = 2
CHANNEL_DOLP = 3

#: Rescales uint8 L/a*/b* onto DoLP's native [0.0, 1.0] float range so no
#: stacked channel dominates the others by two orders of magnitude going
#: into the autoencoder (ocean-vision-v1-133). This is the uint8 encoding's
#: own native range (255 = 2**8 - 1), not a per-site calibration number --
#: see the module docstring (bead ocean-vision-v1-3hw) for why this was
#: chosen over per-channel z-scoring.
_UINT8_TO_UNIT_SCALE = 255.0


class TensorAssemblyError(RuntimeError):
    """Raised when a window of per-frame channels can't be assembled into a
    valid inference tensor -- a short or long window, planes that don't
    share one shape within a frame, or a non-finite value smuggled in from
    upstream. Per this repo's fail-loud rule, assembly never pads a short
    window or coerces a mismatched plane; it raises."""


@dataclass(frozen=True)
class FrameChannels:
    """One frame's [L, a*, b*, DoLP] planes at native ingress resolution,
    each a (H, W) array. `l`/`a`/`b` are `LabNormalizer.normalize()`'s
    output (uint8); `dolp` is `compute_dolp()`'s output (float64 in
    [0, 1]). All four must share one (H, W) shape -- `assemble()` checks
    this per frame and raises rather than resizing them into agreement.
    """

    l: np.ndarray
    a: np.ndarray
    b: np.ndarray
    dolp: np.ndarray

    @classmethod
    def from_lab_and_dolp(
        cls,
        lab: LabFrame,
        dolp: np.ndarray,
        *,
        glare_masker: Optional[SpecularGlareMasker] = None,
    ) -> "FrameChannels":
        """Compose a `LabNormalizer` output and a `compute_dolp()` map into
        one frame's tensor channels.

        If `glare_masker` is given, its glare mask (spec 4.1,
        ocean-vision-v1-bze) is applied to the L plane before packaging --
        the specular-glare suppression `dolp.py`'s own module docstring
        describes as happening "before tensor assembly" -- without touching
        a*/b* (chromaticity isn't a specular signal the way L is) or DoLP
        itself (DoLP rides into channel 4 unmodified, per that same
        docstring). `glare_masker` is optional and unset by default because
        it carries a per-site calibrated `dolp_threshold`
        (`SpecularGlareMasker`'s own constructor requires one); callers with
        no site config yet get an unsuppressed L rather than a silently
        invented threshold.
        """
        l_plane = lab.l if glare_masker is None else glare_masker.suppress(lab.l, dolp)
        return cls(l=l_plane, a=lab.a, b=lab.b, dolp=dolp)


@dataclass(frozen=True)
class AssembledTensor:
    """The assembled inference tensor plus whether the pinned-memory path
    actually ran.

    `pinned` is False whenever the runtime has no CUDA/accelerator device to
    pin host memory against -- true of this dev container, not necessarily
    true of the Jetson AGX Orin target (spec 5.1). `pin_skipped_reason`
    carries why, so a `pinned=False` is never mistaken for a successful
    zero-copy path on hardware this container cannot produce.
    """

    tensor: torch.Tensor
    pinned: bool
    pin_skipped_reason: Optional[str]


class InferenceTensorAssembler:
    """Assembles a spec-4.2-shaped inference tensor from a window of
    per-frame [L, a*, b*, DoLP] planes.

    `target_height`/`target_width`/`window_frames` default to the spec's
    1080p, 120-frame working shape (spec 2, 4.2). Tests may override them to
    keep synthetic tensors small and fast; production callers should leave
    them at the default.
    """

    def __init__(
        self,
        target_height: int = SPEC_HEIGHT,
        target_width: int = SPEC_WIDTH,
        window_frames: int = SPEC_TEMPORAL_WINDOW_FRAMES,
    ):
        if target_height <= 0 or target_width <= 0:
            raise ValueError(
                f"target_height/target_width must be > 0, got {target_height}x{target_width}"
            )
        if window_frames <= 0:
            raise ValueError(f"window_frames must be > 0, got {window_frames}")
        self._target_height = target_height
        self._target_width = target_width
        self._window_frames = window_frames

    @property
    def output_shape(self) -> tuple[int, int, int, int, int]:
        """The (B, C, T, H, W) shape `assemble()` produces for this
        assembler's configured target resolution and window length."""
        return (
            1,
            SPEC_CHANNELS,
            self._window_frames,
            self._target_height,
            self._target_width,
        )

    def assemble(self, frames: Sequence[FrameChannels], *, pin: bool = True) -> AssembledTensor:
        """Downscale, rescale, stack, and batch a window of per-frame
        channels into the (1, 4, T, H, W) inference tensor.

        L/a*/b* (native uint8 [0, 255]) are divided onto DoLP's native
        [0, 1] float range before stacking (bead ocean-vision-v1-3hw, module
        docstring); all four channels of the returned tensor are in [0, 1].

        Raises `TensorAssemblyError` if `frames` is not exactly
        `window_frames` long, if any frame's four planes don't share one
        (H, W) shape, or if any plane contains non-finite values -- a
        silently truncated window or a NaN riding into the autoencoder is
        exactly the failure this must surface, not absorb.

        If `pin` is True (the default), attempts `torch.Tensor.pin_memory()`
        for the Jetson zero-copy handoff; on a runtime with no
        CUDA/accelerator device (this container) that raises `RuntimeError`,
        which is caught and reported via `AssembledTensor.pinned` /
        `pin_skipped_reason` rather than propagated -- absence of an
        accelerator is an environment fact, not a data-integrity fault.
        """
        if len(frames) != self._window_frames:
            raise TensorAssemblyError(
                f"expected exactly {self._window_frames} frames (spec 2 temporal window), "
                f"got {len(frames)}"
            )

        stacked = np.empty(
            (SPEC_CHANNELS, self._window_frames, self._target_height, self._target_width),
            dtype=np.float32,
        )

        for t, frame in enumerate(frames):
            self._assemble_frame(frame, t, stacked)

        tensor = torch.from_numpy(stacked).unsqueeze(0)  # (C,T,H,W) -> (1,C,T,H,W)

        if not pin:
            return AssembledTensor(tensor=tensor, pinned=False, pin_skipped_reason="pin=False, not requested")

        try:
            pinned_tensor = tensor.pin_memory()
        except RuntimeError as exc:
            return AssembledTensor(tensor=tensor, pinned=False, pin_skipped_reason=str(exc))
        return AssembledTensor(tensor=pinned_tensor, pinned=True, pin_skipped_reason=None)

    def _assemble_frame(self, frame: FrameChannels, t: int, stacked: np.ndarray) -> None:
        planes = {"l": frame.l, "a": frame.a, "b": frame.b, "dolp": frame.dolp}
        shape = frame.l.shape
        for name, plane in planes.items():
            if plane.ndim != 2:
                raise TensorAssemblyError(
                    f"frame {t}: '{name}' plane must be (H, W), got shape {plane.shape}"
                )
            if plane.shape != shape:
                raise TensorAssemblyError(
                    f"frame {t}: planes must share one shape; l is {shape}, {name} is {plane.shape}"
                )
            if not np.all(np.isfinite(plane)):
                raise TensorAssemblyError(
                    f"frame {t}: '{name}' plane contains non-finite (NaN/inf) values"
                )

        # L/a*/b* are rescaled onto DoLP's native [0, 1] range (bead
        # ocean-vision-v1-3hw, module docstring) so no channel enters the
        # autoencoder at up to 255x another's magnitude. Division commutes
        # with INTER_AREA/INTER_LINEAR resampling (both are weighted
        # averages), so scaling after resize vs. before is equivalent up to
        # float rounding; scaling after keeps `_resize` itself scale-agnostic.
        stacked[CHANNEL_L, t] = self._resize(frame.l) / _UINT8_TO_UNIT_SCALE
        stacked[CHANNEL_A, t] = self._resize(frame.a) / _UINT8_TO_UNIT_SCALE
        stacked[CHANNEL_B, t] = self._resize(frame.b) / _UINT8_TO_UNIT_SCALE
        stacked[CHANNEL_DOLP, t] = self._resize(frame.dolp)

    def _resize(self, plane: np.ndarray) -> np.ndarray:
        height, width = plane.shape
        if (height, width) == (self._target_height, self._target_width):
            return plane.astype(np.float32, copy=True)

        # INTER_AREA is the recommended OpenCV choice for shrinking (spec
        # 4.2's real path: 4K ingress -> 1080p); INTER_LINEAR is used
        # otherwise (e.g. small synthetic test planes growing to a small
        # target). This is a resampling-algorithm choice, not a per-site
        # calibration constant.
        input_area = height * width
        target_area = self._target_height * self._target_width
        interpolation = cv2.INTER_AREA if input_area > target_area else cv2.INTER_LINEAR

        return cv2.resize(
            plane.astype(np.float32, copy=False),
            (self._target_width, self._target_height),
            interpolation=interpolation,
        )


def _synthetic_frame_channels(height: int = 32, width: int = 48, seed: int = 0) -> FrameChannels:
    """A synthetic frame's worth of [L, a*, b*, DoLP] planes -- exercises the
    assembly code path, not real Lab/DoLP statistics."""
    rng = np.random.default_rng(seed)
    l_plane = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    a_plane = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    b_plane = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    dolp_plane = rng.uniform(0.0, 1.0, size=(height, width))
    return FrameChannels(l=l_plane, a=a_plane, b=b_plane, dolp=dolp_plane)


def self_check() -> AssembledTensor:
    """Standalone runnable check (also invoked from the unittest suite): a
    small synthetic window assembles into the expected shape/dtype/channel
    order, and the pinned-memory path reports honestly for whatever
    accelerator (or lack of one) this runtime has.

    Raises `AssertionError` on any failure.
    """
    window_frames, height, width = 6, 16, 24
    frames = [_synthetic_frame_channels(64, 96, seed=i) for i in range(window_frames)]
    assembler = InferenceTensorAssembler(
        target_height=height, target_width=width, window_frames=window_frames
    )

    result = assembler.assemble(frames)

    assert result.tensor.shape == (1, SPEC_CHANNELS, window_frames, height, width), (
        f"unexpected tensor shape {tuple(result.tensor.shape)}"
    )
    assert result.tensor.dtype == torch.float32, f"unexpected dtype {result.tensor.dtype}"
    assert result.pinned == result.tensor.is_pinned(), (
        "AssembledTensor.pinned must match the tensor's actual pinned state"
    )
    if not torch.cuda.is_available():
        assert not result.pinned, "pin_memory should not silently succeed with no accelerator"
        assert result.pin_skipped_reason, "a skipped pin must carry a reason"

    return result


if __name__ == "__main__":
    self_check()
    print("tensor_assemble self-check: OK")
