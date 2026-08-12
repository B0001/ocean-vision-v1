"""Dual-sensor frame ingress: capture sources, RGB/NIR synchronization, and
the 120-frame rolling ring buffer (spec 2, 4.1)."""

from .frame import Frame
from .ring_buffer import (
    SPEC_TEMPORAL_WINDOW_FRAMES,
    FrameGapError,
    FrameReorderError,
    RingBuffer,
)
from .source import FrameSource, SyntheticFrameSource
from .sync import (
    DualSensorSynchronizer,
    SensorStarvedError,
    SensorSyncError,
    SyncedFramePair,
)

__all__ = [
    "Frame",
    "FrameSource",
    "SyntheticFrameSource",
    "DualSensorSynchronizer",
    "SensorSyncError",
    "SensorStarvedError",
    "SyncedFramePair",
    "RingBuffer",
    "FrameGapError",
    "FrameReorderError",
    "SPEC_TEMPORAL_WINDOW_FRAMES",
]
