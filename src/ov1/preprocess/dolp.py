"""Linear degree-of-polarization (DoLP) specular glare suppression from
Stokes parameters (spec 4.1).

Sun glint off a wave crest is close to specularly reflected light, which is
strongly linearly polarized; diffuse light scattered back out of open water
is not. DoLP -- the fraction of light intensity that is linearly polarized
-- is therefore a physics-based discriminator between the two, computed from
the 850nm polarized-NIR channel's Stokes parameters and independent of scene
content. This module computes that map and the glare mask derived from it.
`compute_dolp`'s output is also channel 4 of the spec 4.2 inference tensor,
unmodified by the mask -- the mask is applied to *other* channels (e.g. the
`LabNormalizer` L plane) before tensor assembly, which happens in
`ocean-vision-v1-3fy`, not here.

Stokes parameters are derived from four intensity planes sampled at
polarizer orientations 0, 45, 90, 135 degrees -- the standard
division-of-focal-plane arrangement for a linear-polarization sensor. Going
from the 850nm sensor's raw per-pixel-mosaic output to these four
already-demosaiced planes is a hardware-driver concern this repo has not
built yet (`ingress/source.py` notes no real capture backend exists); this
module starts from the four planes, however they arrive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class DolpComputationError(RuntimeError):
    """Raised when Stokes/DoLP computation would produce a non-finite or
    physically-invalid result -- e.g. a zero-intensity pixel across all four
    polarizer channels (a dead pixel or a lens cap, not a real glare
    reading), or a computed DoLP that exceeds 1.0 by more than numerical
    noise (Stokes input that isn't valid linear-polarization data). Per this
    repo's fail-loud rule, this is exactly the "NaN in the flow field" case:
    substituting 0 and continuing would hide a sensor fault behind a
    plausible-looking glare map instead of surfacing it.
    """


@dataclass(frozen=True)
class StokesParameters:
    """S0 (total intensity), S1, S2 linear Stokes components, each an
    (H, W) float64 plane. S3 (circular polarization) is not modelled --
    natural light and specular water reflection are both governed by linear
    polarization, and the sensor is a linear division-of-focal-plane
    polarimeter with no circular-polarization channel."""

    s0: np.ndarray
    s1: np.ndarray
    s2: np.ndarray


def compute_stokes(
    i0: np.ndarray, i45: np.ndarray, i90: np.ndarray, i135: np.ndarray
) -> StokesParameters:
    """Compute S0/S1/S2 from the four 0/45/90/135-degree polarizer intensity
    planes.

    S0 = (I0 + I45 + I90 + I135) / 2 -- averaging both orthogonal-pair
    estimates of total intensity (I0+I90 and I45+I135) rather than using
    only one, so a single noisy channel doesn't skew the total-intensity
    denominator DoLP divides by.
    S1 = I0 - I90, S2 = I45 - I135.

    Raises `ValueError` on shape mismatch or negative intensities, and
    `DolpComputationError` on non-finite input -- coercing either into a
    result would fabricate a Stokes value.
    """
    planes = {"i0": i0, "i45": i45, "i90": i90, "i135": i135}
    shape = i0.shape
    for name, plane in planes.items():
        if plane.ndim != 2:
            raise ValueError(f"{name} must be a (H, W) plane, got shape {plane.shape}")
        if plane.shape != shape:
            raise ValueError(
                f"polarization planes must share one shape; i0 is {shape}, {name} is {plane.shape}"
            )
        if not np.all(np.isfinite(plane)):
            raise DolpComputationError(f"{name} contains non-finite (NaN/inf) values")
        if np.any(plane < 0):
            raise ValueError(f"{name} contains negative intensity values")

    i0f, i45f, i90f, i135f = (p.astype(np.float64) for p in (i0, i45, i90, i135))
    s0 = 0.5 * (i0f + i45f + i90f + i135f)
    s1 = i0f - i90f
    s2 = i45f - i135f
    return StokesParameters(s0=s0, s1=s1, s2=s2)


#: Numerical floor below which S0 (total intensity) is treated as no-signal
#: rather than a valid denominator -- a division-by-near-zero guard, not a
#: spec or calibration number. My guess, chosen the same way homography.py's
#: 1e-9 behind-camera weight threshold was: small enough to only catch
#: genuinely degenerate input, not real dim pixels.
_S0_FLOOR = 1e-6

#: Tolerance for DoLP exceeding its theoretical 1.0 ceiling due to float
#: rounding in S1/S2/S0 before being clipped. My guess, not derived from
#: sensor noise characteristics this container cannot measure.
_DOLP_OVERSHOOT_TOLERANCE = 1e-6


def compute_dolp(stokes: StokesParameters) -> np.ndarray:
    """Linear degree of polarization, DoLP = sqrt(S1^2 + S2^2) / S0, as an
    (H, W) float64 array in [0, 1] at frame resolution (spec 4.1's
    acceptance criterion).

    Raises `ValueError` on shape mismatch, and `DolpComputationError` if any
    pixel's S0 is at or below the numerical floor (DoLP is undefined there,
    not zero) or if the result is non-finite or exceeds 1.0 beyond float
    tolerance (physically invalid Stokes input). Never silently clips a
    grossly out-of-range value into [0, 1] -- only the final, validated
    result is clipped, to absorb the last bit of float noise.
    """
    if stokes.s0.shape != stokes.s1.shape or stokes.s0.shape != stokes.s2.shape:
        raise ValueError(
            f"s0/s1/s2 must share one shape; got s0={stokes.s0.shape}, "
            f"s1={stokes.s1.shape}, s2={stokes.s2.shape}"
        )
    if np.any(stokes.s0 < _S0_FLOOR):
        raise DolpComputationError(
            f"S0 (total intensity) is at or below the numerical floor {_S0_FLOOR} "
            "for at least one pixel -- DoLP is undefined there (zero-signal or "
            "dead-pixel condition), not a valid glare reading"
        )

    dolp = np.sqrt(stokes.s1**2 + stokes.s2**2) / stokes.s0

    if not np.all(np.isfinite(dolp)):
        raise DolpComputationError("DoLP computation produced non-finite values")
    if np.any(dolp > 1.0 + _DOLP_OVERSHOOT_TOLERANCE):
        raise DolpComputationError(
            "DoLP exceeded 1.0 beyond numerical tolerance -- input Stokes "
            "parameters are not valid linear-polarization data"
        )

    return np.clip(dolp, 0.0, 1.0)


@dataclass(frozen=True)
class GlareMask:
    """Per-pixel specular-glare classification derived from a DoLP map.

    `glare` is a boolean (H, W) array, True where DoLP met or exceeded the
    calibrated threshold. `suppression` is a (H, W) float64 array, 0.0 on
    glare pixels and 1.0 elsewhere, meant to be multiplied against a
    companion channel (e.g. the L plane from `LabNormalizer`) to suppress
    glare before tensor assembly. This is a hard cut, not a soft ramp: spec
    4.1 calls this a "mask", and a soft falloff would need its own
    calibrated curve nobody has specified.
    """

    glare: np.ndarray
    suppression: np.ndarray


class SpecularGlareMasker:
    """Builds a glare mask from a DoLP map using a per-site DoLP threshold.

    `dolp_threshold` is a calibration knob, not an algorithm constant: how
    strongly polarized specular sun-glint off wave crests reads at a given
    site depends on sun angle, water surface state, and the 850nm channel's
    own gain -- properties of an installation this container has no footage
    to calibrate against. There is no built-in default; callers must supply
    a value from a site config, tuned against that site's own baseline
    footage -- the same pattern `LabNormalizer`'s `clip_limit` /
    `tile_grid_size` already establish in this package.
    """

    def __init__(self, dolp_threshold: float):
        if not (0.0 < dolp_threshold <= 1.0):
            raise ValueError(f"dolp_threshold must be in (0, 1], got {dolp_threshold}")
        self._dolp_threshold = dolp_threshold

    @property
    def dolp_threshold(self) -> float:
        return self._dolp_threshold

    def mask(self, dolp: np.ndarray) -> GlareMask:
        """Build a `GlareMask` from a DoLP map.

        Raises `ValueError` if `dolp` is not 2D or contains values outside
        [0, 1] -- an out-of-range DoLP means the caller skipped
        `compute_dolp`'s own validation, and this must not silently
        renormalize it.
        """
        if dolp.ndim != 2:
            raise ValueError(f"expected a (H, W) DoLP map, got shape {dolp.shape}")
        if np.any(dolp < 0.0) or np.any(dolp > 1.0):
            raise ValueError("dolp values must be in [0, 1]")

        glare = dolp >= self._dolp_threshold
        suppression = np.where(glare, 0.0, 1.0)
        return GlareMask(glare=glare, suppression=suppression)

    def suppress(self, plane: np.ndarray, dolp: np.ndarray) -> np.ndarray:
        """Apply the glare mask to a companion channel plane, zeroing pixels
        classified as specular glare. Returns a new array; `plane` is never
        mutated in place.

        Raises `ValueError` if `plane`'s shape doesn't match `dolp`'s.
        """
        if plane.shape != dolp.shape:
            raise ValueError(
                f"plane shape {plane.shape} must match dolp shape {dolp.shape}"
            )
        glare_mask = self.mask(dolp)
        return (plane.astype(np.float64) * glare_mask.suppression).astype(plane.dtype)


def _synthetic_polarized_frame(height: int = 64, width: int = 64):
    """Four synthetic 0/45/90/135-degree polarization intensity planes: a
    uniform, unpolarized open-water baseline with a bright, strongly
    polarized circular patch standing in for specular sun-glint off a wave
    crest. This is synthetic data, not a real midday glare clip -- it
    exercises the code path, it does not validate against real sun-glint
    statistics.

    Returns `(i0, i45, i90, i135, glare_region)` where `glare_region` is the
    boolean ground-truth mask of the synthetic glare patch.
    """
    y, x = np.mgrid[0:height, 0:width]
    radius = min(height, width) * 0.15
    glare_region = (x - width * 0.75) ** 2 + (y - height * 0.25) ** 2 < radius**2

    # Baseline: all four planes equal -> S1=S2=0 -> DoLP=0 (unpolarized).
    # Glare patch: strongly imbalanced I0/I90 -> high DoLP.
    i0 = np.where(glare_region, 200.0, 100.0)
    i45 = np.where(glare_region, 110.0, 100.0)
    i90 = np.where(glare_region, 20.0, 100.0)
    i135 = np.where(glare_region, 110.0, 100.0)
    return i0, i45, i90, i135, glare_region


def self_check() -> GlareMask:
    """Standalone runnable check (also invoked from the unittest suite): the
    synthetic unpolarized baseline reads ~0 DoLP, the synthetic glare patch
    reads high DoLP, the resulting mask exactly matches the synthetic glare
    region at a mid-range threshold, and `suppress()` zeroes only the glare
    pixels in a companion plane.

    Raises `AssertionError` on any failure.
    """
    i0, i45, i90, i135, glare_region = _synthetic_polarized_frame()
    stokes = compute_stokes(i0, i45, i90, i135)
    dolp = compute_dolp(stokes)

    assert dolp.shape == i0.shape, "DoLP map must match frame resolution"
    assert np.all(dolp >= 0.0) and np.all(dolp <= 1.0), "DoLP must be in [0, 1]"
    assert np.allclose(dolp[~glare_region], 0.0), "unpolarized baseline must read ~0 DoLP"
    assert np.all(dolp[glare_region] > 0.7), "synthetic glare patch must read high DoLP"

    masker = SpecularGlareMasker(dolp_threshold=0.5)
    mask = masker.mask(dolp)
    assert np.array_equal(mask.glare, glare_region), (
        "glare mask must match the synthetic glare region exactly at this threshold"
    )

    luminance = np.full(i0.shape, 200.0)
    suppressed = masker.suppress(luminance, dolp)
    assert np.all(suppressed[glare_region] == 0.0), "glare pixels must be suppressed to zero"
    assert np.all(suppressed[~glare_region] == 200.0), "non-glare pixels must pass through unmodified"

    return mask


if __name__ == "__main__":
    self_check()
    print("dolp self-check: OK")
