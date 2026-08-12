"""Per-sensor frame sources.

`FrameSource` is the boundary between hardware capture and the rest of the
ingress pipeline. A real installation plugs in a capture backend (e.g. V4L2
for the RGB optical sensor, a polarized-NIR driver for the 850nm channel);
this container has no camera and no GPU, so no such backend is implemented
here -- see the ingress bead handoff for what remains unbuilt. Tests and local
development drive the pipeline with `SyntheticFrameSource` instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional

import numpy as np

from .frame import Frame


class FrameSource(ABC):
    """A single sensor's frame stream, read one frame at a time."""

    sensor_id: str

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """Return the next frame in capture order, or None if the source is
        exhausted (end of a recorded clip; a live camera source should block
        or raise instead of ever returning None)."""
        raise NotImplementedError


class SyntheticFrameSource(FrameSource):
    """Replays a fixed, in-memory sequence of (timestamp_ns, data) frames.

    For tests and local development only -- there is no real camera to read
    from in this environment. `frame_index` is assigned sequentially starting
    at 0, modelling a sensor's own hardware frame counter.
    """

    def __init__(self, sensor_id: str, frames: Iterable[tuple[int, np.ndarray]]):
        self.sensor_id = sensor_id
        self._frames = list(frames)
        self._cursor = 0

    def read(self) -> Optional[Frame]:
        if self._cursor >= len(self._frames):
            return None
        timestamp_ns, data = self._frames[self._cursor]
        frame = Frame(
            sensor_id=self.sensor_id,
            frame_index=self._cursor,
            timestamp_ns=timestamp_ns,
            data=data,
        )
        self._cursor += 1
        return frame

    def __len__(self) -> int:
        return len(self._frames)
