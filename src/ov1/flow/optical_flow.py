"""Sparse optical flow velocity field v(x,y,t) (spec 3, 4.3; bead
ocean-vision-v1-bag).

Tracks foam tracer particles from breaking waves across consecutive L-channel
(luminance) frames using pyramidal Lucas-Kanade (`cv2.calcOpticalFlowPyrLK`),
per the bead's own description. The per-window output feeds both inference
heads: the reconstruction-error autoencoder (spec 4.2) and the rip-current
classifier (spec 4.3), which is why velocity is reported in metres/second via
a site `Homography` rather than raw pixels/second -- spec 4.3's 0.5 m/s
threshold is a physical quantity, not a pixel count, and this module leaves
that threshold itself to the classifier (bead ocean-vision-v1-xe3), which is
where a per-site `tau`-style knob belongs.

`ov1.preprocess.lab.LabNormalizer` passes `L` through unmodified for exactly
this reason: this tracker consumes its `l` plane directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .homography import Homography


class NoTrackablePointsError(RuntimeError):
    """Raised when corner detection finds nothing to track -- a blank or
    featureless frame (e.g. flat calm water with no foam, or a sensor fault
    producing a uniform image) rather than a legitimate zero-particle
    velocity field. Silently returning an empty field would let the alarm
    path go quiet without any indication why."""

    def __init__(self, context: str):
        super().__init__(f"no trackable corners found ({context})")


class FlowFieldError(RuntimeError):
    """Raised when the flow field itself is invalid -- NaN/Inf pixel
    positions out of `calcOpticalFlowPyrLK`, or every tracked point lost in
    one step. Never substituted with a default and passed downstream."""


@dataclass(frozen=True)
class VelocityField:
    """The tracked-point velocity field for one frame-to-frame step.

    `positions_px` are the tracked points' *current*-frame pixel positions,
    (N, 2) float64. `velocity_mps` is the corresponding (N, 2) (vx, vy) in
    metres/second on the site's calibrated water plane, computed from the
    homography-mapped displacement over `dt_s`. Rows correspond 1:1.
    """

    positions_px: np.ndarray
    velocity_mps: np.ndarray
    dt_s: float

    def __post_init__(self) -> None:
        if self.positions_px.shape != self.velocity_mps.shape:
            raise ValueError(
                f"positions_px {self.positions_px.shape} and velocity_mps "
                f"{self.velocity_mps.shape} must match"
            )
        if self.positions_px.ndim != 2 or self.positions_px.shape[1] != 2:
            raise ValueError(f"expected (N, 2) arrays, got {self.positions_px.shape}")

    def __len__(self) -> int:
        return len(self.positions_px)


def _detect_corners(
    l_frame: np.ndarray,
    max_corners: int,
    quality_level: float,
    min_distance: float,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Returns (N, 2) float32 corner points, N possibly 0 (never raises --
    callers decide whether zero corners is an error in their context)."""
    corners = cv2.goodFeaturesToTrack(
        l_frame,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        mask=mask,
    )
    if corners is None:
        return np.empty((0, 2), dtype=np.float32)
    return corners.reshape(-1, 2)


class SparseFlowTracker:
    """Lucas-Kanade sparse tracker producing a per-step velocity field.

    `homography` and `frame_period_s` are site/hardware calibration and are
    required -- there is no default site or default frame rate. The corner-
    detection and Lucas-Kanade parameters (`max_corners`, `quality_level`,
    `min_distance`, `win_size`, `max_pyramid_level`, `re_seed_fraction`) are
    algorithm tuning constants, not per-site calibration, so they carry
    conventional OpenCV defaults; callers may still override them.

    Usage: call `seed()` once with the first frame's L-channel, then `step()`
    once per subsequent frame. Drift-tolerant re-seeding happens inside
    `step()`: whenever the count of successfully-tracked points falls below
    `re_seed_fraction * max_corners`, fresh corners are detected (excluding a
    `min_distance` ring around every still-tracked point) and folded in so
    coverage recovers without resetting points that are still being tracked
    cleanly.
    """

    def __init__(
        self,
        homography: Homography,
        frame_period_s: float,
        max_corners: int = 200,
        quality_level: float = 0.01,
        min_distance: float = 7.0,
        win_size: tuple[int, int] = (15, 15),
        max_pyramid_level: int = 2,
        re_seed_fraction: float = 0.5,
    ):
        if frame_period_s <= 0:
            raise ValueError(f"frame_period_s must be > 0, got {frame_period_s}")
        if max_corners <= 0:
            raise ValueError(f"max_corners must be > 0, got {max_corners}")
        if not 0.0 < quality_level < 1.0:
            raise ValueError(f"quality_level must be in (0, 1), got {quality_level}")
        if min_distance <= 0:
            raise ValueError(f"min_distance must be > 0, got {min_distance}")
        if not 0.0 < re_seed_fraction <= 1.0:
            raise ValueError(f"re_seed_fraction must be in (0, 1], got {re_seed_fraction}")

        self._homography = homography
        self._frame_period_s = frame_period_s
        self._max_corners = max_corners
        self._quality_level = quality_level
        self._min_distance = min_distance
        self._win_size = win_size
        self._max_pyramid_level = max_pyramid_level
        self._re_seed_threshold = re_seed_fraction * max_corners

        self._prev_frame: np.ndarray | None = None
        self._prev_points: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self._last_re_seed_count = 0

    @property
    def tracked_point_count(self) -> int:
        return len(self._prev_points)

    @property
    def last_re_seed_count(self) -> int:
        """How many fresh points the most recent `step()` call added.
        Exposed for tests and telemetry, not used in control flow."""
        return self._last_re_seed_count

    def _validate_frame(self, l_frame: np.ndarray) -> None:
        if l_frame.ndim != 2:
            raise ValueError(f"expected a single-channel (H, W) L-plane frame, got shape {l_frame.shape}")
        if l_frame.dtype != np.uint8:
            raise ValueError(f"expected a uint8 L-plane frame, got dtype {l_frame.dtype}")
        if self._prev_frame is not None and l_frame.shape != self._prev_frame.shape:
            raise ValueError(
                f"frame shape changed mid-track: {self._prev_frame.shape} -> {l_frame.shape}"
            )

    def seed(self, l_frame: np.ndarray) -> None:
        """(Re)start tracking from scratch on `l_frame`'s L-channel plane.

        Raises `NoTrackablePointsError` if no corners are found -- a
        featureless frame is a data problem to surface, not a valid
        zero-point start state.
        """
        self._validate_frame(l_frame)
        corners = _detect_corners(l_frame, self._max_corners, self._quality_level, self._min_distance)
        if len(corners) == 0:
            raise NoTrackablePointsError("seed frame has no detectable corners")

        self._prev_frame = l_frame
        self._prev_points = corners.astype(np.float32)
        self._last_re_seed_count = 0

    def _exclusion_mask(self, shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
        mask = np.full(shape, 255, dtype=np.uint8)
        radius = int(round(self._min_distance))
        for x, y in points:
            cv2.circle(mask, (int(round(x)), int(round(y))), radius, 0, -1)
        return mask

    def step(self, l_frame: np.ndarray) -> VelocityField:
        """Advance tracking by one frame and return the velocity field.

        Raises `RuntimeError` if `seed()` has not been called yet,
        `FlowFieldError` if every tracked point is lost or the optical flow
        result contains non-finite pixel positions.
        """
        if self._prev_frame is None:
            raise RuntimeError("step() called before seed(); call seed() with the first frame")
        self._validate_frame(l_frame)

        if len(self._prev_points) == 0:
            raise FlowFieldError("no points under track entering this step (prior step lost all points)")

        next_points, status, _err = cv2.calcOpticalFlowPyrLK(
            self._prev_frame,
            l_frame,
            self._prev_points.reshape(-1, 1, 2),
            None,
            winSize=self._win_size,
            maxLevel=self._max_pyramid_level,
        )
        next_points = next_points.reshape(-1, 2)
        tracked_mask = status.reshape(-1) == 1

        prev_tracked = self._prev_points[tracked_mask]
        curr_tracked = next_points[tracked_mask]

        if len(curr_tracked) == 0:
            raise FlowFieldError("all tracked points lost in this step")
        if not np.isfinite(curr_tracked).all():
            raise FlowFieldError("non-finite pixel position in optical flow output")

        prev_metres = self._homography.pixel_to_metres(prev_tracked)
        curr_metres = self._homography.pixel_to_metres(curr_tracked)
        velocity_mps = (curr_metres - prev_metres) / self._frame_period_s

        if len(curr_tracked) < self._re_seed_threshold:
            exclusion_mask = self._exclusion_mask(l_frame.shape, curr_tracked)
            budget = self._max_corners - len(curr_tracked)
            fresh = _detect_corners(
                l_frame, budget, self._quality_level, self._min_distance, mask=exclusion_mask
            )
            self._last_re_seed_count = len(fresh)
            self._prev_points = np.concatenate([curr_tracked, fresh], axis=0).astype(np.float32)
        else:
            self._last_re_seed_count = 0
            self._prev_points = curr_tracked.astype(np.float32)

        self._prev_frame = l_frame

        return VelocityField(positions_px=curr_tracked, velocity_mps=velocity_mps, dt_s=self._frame_period_s)


def _synthetic_drift_clip(
    n_frames: int = 6,
    height: int = 120,
    width: int = 160,
    dx_px_per_frame: float = 5.0,
    dy_px_per_frame: float = 0.0,
    seed: int = 42,
) -> list[np.ndarray]:
    """A textured synthetic clip translating at a fixed, known pixel offset
    per frame -- stands in for foam tracers drifting in a known current
    direction. This is synthetic motion, not ocean footage: it exercises
    the tracker's mechanics, it does not validate real foam-tracer behavior.
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(height, width), dtype=np.uint8)
    frames = []
    for i in range(n_frames):
        shift = np.array(
            [[1, 0, dx_px_per_frame * i], [0, 1, dy_px_per_frame * i]], dtype=np.float32
        )
        frames.append(cv2.warpAffine(base, shift, (width, height), borderMode=cv2.BORDER_WRAP))
    return frames


def self_check() -> VelocityField:
    """Standalone runnable check (also invoked from the unittest suite):
    tracks a synthetic clip drifting at a known constant pixel velocity and
    confirms the recovered metre-space velocity direction and rough
    magnitude match what the known pixel drift and homography scale predict.

    Raises AssertionError on any failure.
    """
    dx_px_per_frame = 5.0
    frame_period_s = 1.0 / 30.0
    metres_per_pixel = 0.05  # arbitrary synthetic site scale for this check

    homography = Homography(
        matrix=np.array(
            [[metres_per_pixel, 0, 0], [0, metres_per_pixel, 0], [0, 0, 1]], dtype=np.float64
        )
    )
    tracker = SparseFlowTracker(homography=homography, frame_period_s=frame_period_s)

    clip = _synthetic_drift_clip(dx_px_per_frame=dx_px_per_frame)
    tracker.seed(clip[0])

    expected_vx_mps = metres_per_pixel * dx_px_per_frame / frame_period_s
    for frame in clip[1:]:
        field = tracker.step(frame)
        assert len(field) > 0, "velocity field must not be empty"
        mean_v = field.velocity_mps.mean(axis=0)
        assert mean_v[0] > 0, f"expected seaward (+x) drift, got mean vx={mean_v[0]}"
        assert abs(mean_v[0] - expected_vx_mps) < 0.15 * expected_vx_mps, (
            f"mean vx {mean_v[0]} far from expected {expected_vx_mps} for known pixel drift"
        )
        assert abs(mean_v[1]) < 0.1 * expected_vx_mps, f"unexpected cross-current vy={mean_v[1]}"

    return field


if __name__ == "__main__":
    self_check()
    print("optical_flow self-check: OK")
