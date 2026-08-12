"""CIE Lab depth-attenuation normalization (spec 4.1).

Standard RGB fails underwater because long wavelengths attenuate rapidly with
depth (e^(-eta*d)). The pipeline instead converts each frame to CIE Lab and
applies dynamic contrast stretching *strictly* to the a*/b* chromaticity
planes to compensate. L is passed through unmodified so luminance-based
optical flow downstream stays valid -- stretching L would distort the motion
signal the flow estimator (bead ocean-vision-v1-bag) depends on.

Frame data is assumed to arrive as (H, W, 3) uint8 in BGR channel order (the
OpenCV convention the bead's own description names: `cv2.cvtColor(...,
COLOR_BGR2Lab)`). Nothing upstream in this repo fixes sensor channel order
yet -- if the real RGB driver delivers RGB order instead, the caller must
convert before calling `normalize()`, or this silently swaps the red and blue
attenuation compensation.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LabFrame:
    """L, a*, b* planes for one frame, each (H, W) uint8.

    `l` is bit-identical to `cv2.cvtColor(frame, COLOR_BGR2Lab)`'s L plane --
    never stretched. `a` and `b` have had CLAHE applied.
    """

    l: np.ndarray
    a: np.ndarray
    b: np.ndarray


class LabNormalizer:
    """Converts a BGR frame to Lab and CLAHE-stretches the a*/b* planes only.

    `clip_limit` and `tile_grid_size` are CLAHE calibration knobs, not
    algorithm constants: how much stretch is appropriate depends on the
    site's water turbidity (the eta in e^(-eta*d)), which this container has
    no footage to calibrate against. There is no built-in default -- callers
    must supply values from a site config, tuned against that site's own
    baseline footage.
    """

    def __init__(self, clip_limit: float, tile_grid_size: tuple[int, int]):
        if clip_limit <= 0:
            raise ValueError(f"clip_limit must be > 0, got {clip_limit}")
        if len(tile_grid_size) != 2 or any(d <= 0 for d in tile_grid_size):
            raise ValueError(
                f"tile_grid_size must be a pair of positive ints, got {tile_grid_size}"
            )
        self._clip_limit = clip_limit
        self._tile_grid_size = tile_grid_size

    def normalize(self, bgr_frame: np.ndarray) -> LabFrame:
        """Convert one BGR frame to Lab and CLAHE-stretch a*/b* only.

        Raises `ValueError` on any shape/dtype mismatch rather than
        coercing -- a silently reshaped or recast frame is exactly the kind
        of substituted-default failure this pipeline must not produce.
        """
        if bgr_frame.ndim != 3 or bgr_frame.shape[2] != 3:
            raise ValueError(f"expected an (H, W, 3) BGR frame, got shape {bgr_frame.shape}")
        if bgr_frame.dtype != np.uint8:
            raise ValueError(f"expected a uint8 frame, got dtype {bgr_frame.dtype}")

        lab = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2Lab)
        l_plane, a_plane, b_plane = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=self._clip_limit, tileGridSize=self._tile_grid_size)
        a_stretched = clahe.apply(a_plane)
        b_stretched = clahe.apply(b_plane)

        return LabFrame(l=l_plane, a=a_stretched, b=b_stretched)


def _synthetic_turbid_frame(height: int = 64, width: int = 64) -> np.ndarray:
    """A synthetic BGR frame standing in for turbid, depth-attenuated water:
    each channel is squeezed into a 3-value range, giving the resulting a*/b*
    planes a narrow, skewed histogram the way real long-wavelength
    attenuation would. This is synthetic data, not ocean footage -- it
    exercises the code path, it does not validate against real turbidity.
    """
    y, x = np.mgrid[0:height, 0:width]
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 0] = 120 + (x % 3)  # B: attenuates least, still squeezed here
    frame[..., 1] = 100 + (y % 3)  # G
    frame[..., 2] = 90 + ((x + y) % 3)  # R: attenuates most in real water
    return frame


def self_check() -> LabFrame:
    """Standalone runnable check (also invoked from the unittest suite):
    L is bit-identical to the unstretched conversion, and CLAHE visibly
    widens the a*/b* histograms on synthetic turbid footage while leaving L
    untouched -- i.e. the mutation is channel-selective, not global.

    Raises AssertionError on any failure.
    """
    frame = _synthetic_turbid_frame()
    normalizer = LabNormalizer(clip_limit=2.0, tile_grid_size=(8, 8))
    result = normalizer.normalize(frame)

    unstretched_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    unstretched_l, unstretched_a, unstretched_b = cv2.split(unstretched_lab)

    assert np.array_equal(result.l, unstretched_l), "L plane must be bit-identical to unstretched conversion"

    a_range_before = int(unstretched_a.max()) - int(unstretched_a.min())
    a_range_after = int(result.a.max()) - int(result.a.min())
    b_range_before = int(unstretched_b.max()) - int(unstretched_b.min())
    b_range_after = int(result.b.max()) - int(result.b.min())
    assert a_range_after > a_range_before, (
        f"a* histogram not de-skewed: range {a_range_before} -> {a_range_after}"
    )
    assert b_range_after > b_range_before, (
        f"b* histogram not de-skewed: range {b_range_before} -> {b_range_after}"
    )

    assert not np.array_equal(result.a, unstretched_a), "a* plane was not mutated"
    assert not np.array_equal(result.b, unstretched_b), "b* plane was not mutated"

    return result


if __name__ == "__main__":
    self_check()
    print("lab_normalize self-check: OK")
