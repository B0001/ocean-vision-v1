"""End-to-end: two sensor sources -> synchronizer -> ring buffer.

Covers this bead's acceptance criteria together: two synchronized streams
decoded, a sustained 120-frame window with a frame drop counter, and <1
frame skew between sensors for every buffered pair. Runs entirely on
synthetic in-memory tensors -- no camera, no ocean footage, no claim about
real-world frame rate or skew is made here (see handoff).
"""

import unittest

import numpy as np

from ov1.ingress import (
    DualSensorSynchronizer,
    RingBuffer,
    SensorStarvedError,
    SensorSyncError,
    SyntheticFrameSource,
)
from ov1.ingress.ring_buffer import FrameGapError

FRAME_RATE_HZ = 60.0
PERIOD_NS = round(1e9 / FRAME_RATE_HZ)
WINDOW_FRAMES = 120


def _px(i):
    return np.full((4, 4), i % 256, dtype=np.uint8)


def _run_pipeline(rgb_frames, nir_frames, capacity=WINDOW_FRAMES):
    rgb = SyntheticFrameSource("rgb_optical", rgb_frames)
    nir = SyntheticFrameSource("nir_850nm", nir_frames)
    sync = DualSensorSynchronizer(rgb, nir, nominal_frame_rate_hz=FRAME_RATE_HZ)
    buf = RingBuffer(capacity=capacity)

    while not buf.is_full:
        try:
            pair = sync.next_pair()
        except SensorStarvedError:
            break
        except SensorSyncError:
            continue
        try:
            buf.push(pair)
        except FrameGapError:
            pass  # already reflected in buf.dropped_frame_count

    return sync, buf


class TestIngressPipeline(unittest.TestCase):
    def test_clean_streams_sustain_a_120_frame_window_with_no_drops(self):
        n = WINDOW_FRAMES + 5
        rgb_frames = [(i * PERIOD_NS, _px(i)) for i in range(n)]
        nir_frames = [(i * PERIOD_NS + 500, _px(i)) for i in range(n)]  # sub-frame jitter

        sync, buf = _run_pipeline(rgb_frames, nir_frames)

        self.assertTrue(buf.is_full)
        self.assertEqual(len(buf), WINDOW_FRAMES)
        self.assertEqual(sync.dropped_frame_count, 0)
        self.assertEqual(buf.dropped_frame_count, 0)
        for pair in buf.window():
            self.assertLess(abs(pair.skew_ns), sync.skew_tolerance_ns)

    def test_window_still_sustains_120_frames_after_a_missing_nir_frame(self):
        n = WINDOW_FRAMES + 5
        rgb_frames = [(i * PERIOD_NS, _px(i)) for i in range(n)]
        nir_timestamps = [i * PERIOD_NS for i in range(n)]
        del nir_timestamps[5]  # one NIR frame is physically never delivered
        nir_timestamps.append(nir_timestamps[-1] + PERIOD_NS)  # keep stream lengths equal
        nir_frames = [(t, _px(i)) for i, t in enumerate(nir_timestamps)]

        sync, buf = _run_pipeline(rgb_frames, nir_frames)

        self.assertTrue(buf.is_full)
        self.assertEqual(len(buf), WINDOW_FRAMES)
        self.assertEqual(sync.dropped_frame_count, 1)
        for pair in buf.window():
            self.assertLess(abs(pair.skew_ns), sync.skew_tolerance_ns)


if __name__ == "__main__":
    unittest.main()
