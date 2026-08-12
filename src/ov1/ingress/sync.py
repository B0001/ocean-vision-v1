"""Hardware-timestamp alignment of the RGB optical and 850nm NIR streams.

Spec 4.1 requires the two co-axial sensors to be treated as one aligned
stream; the bead's acceptance criteria fix the tolerance at under one frame
period of skew. Every desync is raised, never absorbed silently -- a caller
that ignores the exception stops receiving frames rather than receiving
mismatched ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .frame import Frame
from .source import FrameSource


class SensorSyncError(RuntimeError):
    """Raised when the RGB and NIR frames on hand differ by more than the
    configured skew tolerance. The lagging sensor's frame has been counted as
    dropped; call `next_pair()` again to advance the lagging source and retry."""

    def __init__(self, lagging_sensor: str, skew_ns: int, tolerance_ns: int):
        self.lagging_sensor = lagging_sensor
        self.skew_ns = skew_ns
        self.tolerance_ns = tolerance_ns
        super().__init__(
            f"sensor skew {skew_ns} ns exceeds tolerance {tolerance_ns} ns "
            f"({lagging_sensor} is lagging)"
        )


class SensorStarvedError(RuntimeError):
    """Raised when one sensor's source is exhausted while the other still has
    frames -- a real capture pipeline never returns None, so this signals a
    genuine feed loss rather than end-of-clip in a live deployment."""

    def __init__(self, starved_sensor: str):
        self.starved_sensor = starved_sensor
        super().__init__(f"sensor '{starved_sensor}' produced no frame")


@dataclass(frozen=True)
class SyncedFramePair:
    """One time-aligned (RGB, NIR) frame pair."""

    frame_index: int
    rgb: Frame
    nir: Frame
    skew_ns: int


class DualSensorSynchronizer:
    """Pairs frames from the RGB and NIR sources within a skew tolerance.

    `nominal_frame_rate_hz` and `max_skew_frames` are properties of the
    installation's cameras and mounting, not algorithm constants -- callers
    must supply them explicitly; there is no built-in default.
    """

    def __init__(
        self,
        rgb_source: FrameSource,
        nir_source: FrameSource,
        nominal_frame_rate_hz: float,
        max_skew_frames: float = 1.0,
    ):
        if nominal_frame_rate_hz <= 0:
            raise ValueError("nominal_frame_rate_hz must be > 0")
        if max_skew_frames <= 0:
            raise ValueError("max_skew_frames must be > 0")

        self._rgb = rgb_source
        self._nir = nir_source
        self._frame_period_ns = round(1e9 / nominal_frame_rate_hz)
        self._skew_tolerance_ns = round(self._frame_period_ns * max_skew_frames)
        self._dropped_frame_count = 0

        self._pending_rgb: Optional[Frame] = None
        self._pending_nir: Optional[Frame] = None

    @property
    def skew_tolerance_ns(self) -> int:
        return self._skew_tolerance_ns

    @property
    def dropped_frame_count(self) -> int:
        return self._dropped_frame_count

    def next_pair(self) -> SyncedFramePair:
        """Return the next time-aligned frame pair.

        Raises `SensorSyncError` if the two sensors' next frames fall outside
        the skew tolerance (the lagging sensor's frame is dropped and counted;
        call again to retry with the other sensor advanced), or
        `SensorStarvedError` if a source has no frame to give.
        """
        rgb = self._pending_rgb if self._pending_rgb is not None else self._rgb.read()
        nir = self._pending_nir if self._pending_nir is not None else self._nir.read()
        self._pending_rgb = None
        self._pending_nir = None

        if rgb is None:
            raise SensorStarvedError(self._rgb.sensor_id)
        if nir is None:
            raise SensorStarvedError(self._nir.sensor_id)

        skew_ns = nir.timestamp_ns - rgb.timestamp_ns
        if abs(skew_ns) >= self._skew_tolerance_ns:
            self._dropped_frame_count += 1
            if skew_ns > 0:
                # NIR is ahead; RGB is lagging and gets dropped this round.
                self._pending_nir = nir
                lagging_sensor = self._rgb.sensor_id
            else:
                self._pending_rgb = rgb
                lagging_sensor = self._nir.sensor_id
            raise SensorSyncError(
                lagging_sensor=lagging_sensor,
                skew_ns=skew_ns,
                tolerance_ns=self._skew_tolerance_ns,
            )

        return SyncedFramePair(
            frame_index=rgb.frame_index,
            rgb=rgb,
            nir=nir,
            skew_ns=skew_ns,
        )
