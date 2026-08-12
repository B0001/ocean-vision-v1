"""Single-sensor frame representation shared by every ingress component."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Frame:
    """One captured frame from a single sensor.

    `timestamp_ns` is a hardware-sourced capture timestamp (e.g. a V4L2 buffer
    timestamp or PTP-disciplined clock reading) on a monotonic clock shared by
    both sensors in an installation -- it is not wall-clock time and is not
    assigned by software after the fact. `frame_index` is that sensor's own
    monotonically increasing hardware frame counter, used to detect gaps
    (dropped frames) independently of any downstream synchronization.
    """

    sensor_id: str
    frame_index: int
    timestamp_ns: int
    data: np.ndarray

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError(f"frame_index must be >= 0, got {self.frame_index}")
        if self.timestamp_ns < 0:
            raise ValueError(f"timestamp_ns must be >= 0, got {self.timestamp_ns}")
