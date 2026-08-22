"""Rip current classifier -- Head B of the dual-head inference model (spec
3's pipeline diagram: "G[Head B: Hydrodynamic Anomaly Classifier]", spec 4.3;
bead ocean-vision-v1-xe3).

Spec 4.3's rule, verbatim: if the tracked foam-tracer velocity field
v(x,y,t) points seaward with a sustained magnitude |v| > 0.5 m/s across a
spatial channel narrower than 15 m, the region is an active rip current
hazard. This module turns each per-step
`ov1.flow.optical_flow.VelocityField` (spec 3, bead ocean-vision-v1-bag) into
that classification. 0.5 m/s and 15 m are the spec's own numbers -- unlike
`tau_drowning`, section 4.3 states them as part of the algorithm's physical
definition of "rip current" rather than as a per-site calibration knob, so
they are kept as overridable defaults instead of a required site-config
input. They are spec *targets*, not measurements -- nothing in this module
has run against real rip footage; see "validated on annotated rip footage"
in the bead's own acceptance criteria, which this session could not do (no
ocean footage in this container).

Two inputs *are* mandatory, per this bead's description, and have no
default -- both are properties of one physical camera installation, not of
the algorithm (see CLAUDE.md's calibration-knobs rule):

- `homography`: the site's pixel -> metre ground-plane mapping (already
  required upstream by `SparseFlowTracker`). Needed here again because
  `VelocityField.positions_px` is pixel space, but channel width is a metre
  quantity.
- `shoreline_normal`: the seaward-pointing unit vector at this installation,
  in the same world-plane metre coordinates the homography produces. There
  is no way to derive "seaward" from pixels alone -- it is surveyed once at
  install time.

Spec 4.3 does not give a number for how long "sustained" means, unlike
section 4.2's explicit "Delta t >= 3.0s" for the drowning trigger. Rather
than guess one, `min_sustained_s` is a required constructor argument with no
default, on the same footing as `tau_drowning`: a per-site/per-deployment
calibration decision, not this module's to make silently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ov1.flow.homography import Homography
from ov1.flow.optical_flow import VelocityField


@dataclass(frozen=True)
class ShorelineNormal:
    """The seaward-pointing unit vector for one installation, in the world-
    plane metre coordinates the site `Homography` maps pixels into (spec
    4.3). Surveyed once at install time against the shoreline's actual
    orientation -- never derived from footage, and there is no default."""

    vector: np.ndarray

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector, dtype=np.float64)
        if vector.shape != (2,):
            raise ValueError(f"shoreline normal must be a (2,) vector, got shape {vector.shape}")
        if not np.isfinite(vector).all():
            raise ValueError("shoreline normal must be finite (no NaN/Inf)")
        norm = float(np.linalg.norm(vector))
        if abs(norm - 1.0) > 1e-6:
            raise ValueError(f"shoreline normal must be unit length, got magnitude {norm}")
        object.__setattr__(self, "vector", vector)

    @property
    def tangent(self) -> np.ndarray:
        """The along-shore unit vector, 90 degrees from seaward. Used to
        measure a candidate channel's cross-current width."""
        return np.array([-self.vector[1], self.vector[0]])


@dataclass(frozen=True)
class RipChannel:
    """One spatially-contiguous group of seaward, above-threshold tracked
    points found in a single `VelocityField` step, that also passed the
    channel-width gate -- a rip current *candidate* at this instant. It only
    becomes an active hazard (`RipCurrentObservation.is_active`) once a
    channel has kept qualifying, step over step, for `min_sustained_s`."""

    centroid_m: np.ndarray
    width_m: float
    mean_speed_mps: float
    point_count: int


@dataclass(frozen=True)
class RipCurrentObservation:
    """The classifier's result for one `update()` call.

    `channels` is empty whenever no group of tracked points this step passed
    all three spec 4.3 gates (seaward direction, magnitude, channel width).
    `sustained_s` is how long, continuously, at least one qualifying channel
    has been present -- reset to 0 the instant a step has none. `is_active`
    is the final alert condition: a qualifying channel present now *and*
    `sustained_s >= min_sustained_s`.
    """

    channels: tuple[RipChannel, ...]
    sustained_s: float
    is_active: bool


class RipCurrentClassifier:
    """Stateful spec 4.3 rip current classifier, driven one `VelocityField`
    step at a time from the same `SparseFlowTracker` output feeding Head A
    (spec 3's pipeline diagram).

    Usage: construct once per site/session with the site's `homography` and
    `shoreline_normal`, then call `update(field)` once per tracked step, in
    order -- `sustained_s` is stateful across calls and a skipped or
    reordered step will misreport how long a hazard has persisted.
    """

    def __init__(
        self,
        homography: Homography,
        shoreline_normal: ShorelineNormal,
        min_sustained_s: float,
        velocity_threshold_mps: float = 0.5,
        max_channel_width_m: float = 15.0,
        cluster_gap_m: float | None = None,
        min_channel_points: int = 2,
    ):
        if min_sustained_s <= 0:
            raise ValueError(f"min_sustained_s must be > 0, got {min_sustained_s}")
        if velocity_threshold_mps <= 0:
            raise ValueError(f"velocity_threshold_mps must be > 0, got {velocity_threshold_mps}")
        if max_channel_width_m <= 0:
            raise ValueError(f"max_channel_width_m must be > 0, got {max_channel_width_m}")
        if min_channel_points < 2:
            # A single point has zero width by definition, which would
            # trivially pass the < max_channel_width_m gate without
            # measuring anything -- two points is the minimum for "width"
            # to mean something.
            raise ValueError(f"min_channel_points must be >= 2, got {min_channel_points}")

        resolved_cluster_gap_m = max_channel_width_m if cluster_gap_m is None else cluster_gap_m
        if resolved_cluster_gap_m <= 0:
            raise ValueError(f"cluster_gap_m must be > 0, got {resolved_cluster_gap_m}")

        self._homography = homography
        self._normal = shoreline_normal.vector
        self._tangent = shoreline_normal.tangent
        self._min_sustained_s = min_sustained_s
        self._velocity_threshold_mps = velocity_threshold_mps
        self._max_channel_width_m = max_channel_width_m
        self._cluster_gap_m = resolved_cluster_gap_m
        self._min_channel_points = min_channel_points

        self._sustained_s = 0.0

    @property
    def sustained_s(self) -> float:
        return self._sustained_s

    def reset(self) -> None:
        """Clear accumulated sustain time, e.g. after a tracking gap the
        caller has independently detected (re-seed, dropped frame)."""
        self._sustained_s = 0.0

    def _cluster_channels(self, positions_m: np.ndarray, speeds_mps: np.ndarray) -> tuple[RipChannel, ...]:
        tangential = positions_m @ self._tangent
        order = np.argsort(tangential)
        tangential_sorted = tangential[order]
        positions_sorted = positions_m[order]
        speeds_sorted = speeds_mps[order]

        gaps = np.diff(tangential_sorted)
        split_points = np.where(gaps > self._cluster_gap_m)[0] + 1
        groups = np.split(np.arange(len(tangential_sorted)), split_points)

        channels = []
        for group in groups:
            if len(group) < self._min_channel_points:
                continue
            group_tangential = tangential_sorted[group]
            width_m = float(group_tangential.max() - group_tangential.min())
            if width_m >= self._max_channel_width_m:
                continue
            channels.append(
                RipChannel(
                    centroid_m=positions_sorted[group].mean(axis=0),
                    width_m=width_m,
                    mean_speed_mps=float(speeds_sorted[group].mean()),
                    point_count=len(group),
                )
            )
        return tuple(channels)

    def update(self, field: VelocityField) -> RipCurrentObservation:
        """Classify one tracked step and advance the sustain timer.

        Raises `ValueError` if `field` carries non-finite values or a
        non-positive `dt_s` -- never substituted with a default, per this
        repo's fail-loud rule for the detection path.
        """
        if field.dt_s <= 0:
            raise ValueError(f"VelocityField.dt_s must be > 0, got {field.dt_s}")
        if not np.isfinite(field.positions_px).all() or not np.isfinite(field.velocity_mps).all():
            raise ValueError("VelocityField contains non-finite positions or velocities")

        if len(field) == 0:
            channels: tuple[RipChannel, ...] = ()
        else:
            speeds_mps = np.linalg.norm(field.velocity_mps, axis=1)
            seaward_component = field.velocity_mps @ self._normal
            qualifies = (seaward_component > 0) & (speeds_mps > self._velocity_threshold_mps)

            if not qualifies.any():
                channels = ()
            else:
                positions_m = self._homography.pixel_to_metres(field.positions_px[qualifies])
                channels = self._cluster_channels(positions_m, speeds_mps[qualifies])

        if channels:
            self._sustained_s += field.dt_s
        else:
            self._sustained_s = 0.0

        is_active = bool(channels) and self._sustained_s >= self._min_sustained_s
        return RipCurrentObservation(channels=channels, sustained_s=self._sustained_s, is_active=is_active)


def self_check() -> RipCurrentObservation:
    """Standalone runnable check (also invoked from the unittest suite): a
    synthetic velocity field with a narrow group of fast seaward points and
    a wide group of slow/alongshore noise, driven for enough steps to cross
    an arbitrary `min_sustained_s`, confirms only the narrow seaward group
    is ever reported and that activation waits for the sustain window.

    Raises AssertionError on any failure.
    """
    homography = Homography(matrix=np.eye(3, dtype=np.float64))  # 1 px = 1 m, self-check only
    shoreline_normal = ShorelineNormal(vector=np.array([1.0, 0.0]))
    dt_s = 1.0 / 30.0
    min_sustained_s = 3 * dt_s

    classifier = RipCurrentClassifier(
        homography=homography,
        shoreline_normal=shoreline_normal,
        min_sustained_s=min_sustained_s,
    )

    rip_positions = np.array([[10.0, 0.0], [10.0, 2.0], [10.0, 4.0]])
    rip_velocity = np.tile(np.array([1.0, 0.0]), (3, 1))  # 1.0 m/s straight seaward
    noise_positions = np.array([[50.0, 0.0], [50.0, 30.0]])
    noise_velocity = np.tile(np.array([0.0, 0.3]), (2, 1))  # alongshore drift, not seaward

    positions_px = np.concatenate([rip_positions, noise_positions], axis=0)
    velocity_mps = np.concatenate([rip_velocity, noise_velocity], axis=0)
    field = VelocityField(positions_px=positions_px, velocity_mps=velocity_mps, dt_s=dt_s)

    observation = None
    for step in range(5):
        observation = classifier.update(field)
        assert len(observation.channels) == 1, f"expected exactly one channel, got {len(observation.channels)}"
        channel = observation.channels[0]
        assert channel.width_m < 15.0, f"channel width {channel.width_m} should be under the 15m gate"
        assert channel.point_count == 3, f"expected the 3 rip points, got {channel.point_count}"
        if step < 2:
            assert not observation.is_active, "should not activate before min_sustained_s elapses"
        else:
            assert observation.is_active, "should be active once sustained_s >= min_sustained_s"

    return observation


if __name__ == "__main__":
    self_check()
    print("rip_current self-check: OK")
