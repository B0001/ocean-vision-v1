"""The 120-frame rolling temporal window that every downstream stage reads
from (spec 2, 4.1; this bead's acceptance criteria).

Spec section 2 states the temporal window as "120 frames" independent of
frame rate, while section 4.1 mandates 4K/60 capture; at 60 FPS that window
is 2s, not the 4s you'd get at the 30 FPS also mentioned in section 2. This
buffer is deliberately frame-count-based, not duration-based, to match the
bead's literal acceptance criterion ("sustained 120-frame window") without
picking a side of that spec inconsistency -- flagged for a human to resolve.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from .sync import SyncedFramePair

#: Spec section 2, "Temporal Window" row. Not a measurement -- an
#: architectural constant fixing the buffer's frame-count capacity.
SPEC_TEMPORAL_WINDOW_FRAMES = 120


class FrameGapError(RuntimeError):
    """Raised after a non-contiguous or out-of-order frame is pushed. The
    buffer still advances (the new frame is appended) so the window keeps
    moving forward -- this raises to make the drop visible, not to halt
    ingestion silently."""

    def __init__(self, last_frame_index: int, pushed_frame_index: int, gap: int):
        self.last_frame_index = last_frame_index
        self.pushed_frame_index = pushed_frame_index
        self.gap = gap
        super().__init__(
            f"frame index jumped from {last_frame_index} to {pushed_frame_index} "
            f"({gap} frame(s) dropped)"
        )


class FrameReorderError(RuntimeError):
    """Raised when a pushed frame's index is not after the last one held --
    this buffer never silently reorders or overwrites history."""

    def __init__(self, last_frame_index: int, pushed_frame_index: int):
        self.last_frame_index = last_frame_index
        self.pushed_frame_index = pushed_frame_index
        super().__init__(
            f"frame index {pushed_frame_index} is not after last held index "
            f"{last_frame_index}"
        )


class RingBuffer:
    """Fixed-capacity rolling window of synchronized (RGB, NIR) frame pairs."""

    def __init__(self, capacity: int = SPEC_TEMPORAL_WINDOW_FRAMES):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._capacity = capacity
        self._frames: deque[SyncedFramePair] = deque(maxlen=capacity)
        self._dropped_frame_count = 0
        self._pushed_frame_count = 0
        self._last_frame_index: Optional[int] = None

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def dropped_frame_count(self) -> int:
        return self._dropped_frame_count

    @property
    def pushed_frame_count(self) -> int:
        return self._pushed_frame_count

    @property
    def is_full(self) -> bool:
        return len(self._frames) == self._capacity

    def __len__(self) -> int:
        return len(self._frames)

    def push(self, pair: SyncedFramePair) -> None:
        """Append a synchronized frame pair to the window.

        Raises `FrameReorderError` (without appending) if `pair` is not after
        the last frame held. Raises `FrameGapError` *after* appending if
        `pair`'s frame_index is non-contiguous with the last one held, so the
        window still advances but the gap is never silent.
        """
        if self._last_frame_index is not None and pair.frame_index <= self._last_frame_index:
            raise FrameReorderError(self._last_frame_index, pair.frame_index)

        gap = 0
        if self._last_frame_index is not None:
            gap = pair.frame_index - self._last_frame_index - 1

        self._frames.append(pair)
        self._pushed_frame_count += 1
        prev_index = self._last_frame_index
        self._last_frame_index = pair.frame_index

        if gap > 0:
            self._dropped_frame_count += gap
            raise FrameGapError(prev_index, pair.frame_index, gap)

    def window(self) -> list[SyncedFramePair]:
        """A snapshot of the frames currently held, oldest first."""
        return list(self._frames)
