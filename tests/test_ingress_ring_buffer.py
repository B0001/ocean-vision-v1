import unittest

import numpy as np

from ov1.ingress import Frame, FrameGapError, FrameReorderError, RingBuffer, SyncedFramePair
from ov1.ingress.ring_buffer import SPEC_TEMPORAL_WINDOW_FRAMES


def _pair(index, skew_ns=0):
    rgb = Frame(sensor_id="rgb", frame_index=index, timestamp_ns=index * 1000, data=np.zeros((1, 1)))
    nir = Frame(sensor_id="nir", frame_index=index, timestamp_ns=index * 1000 + skew_ns, data=np.zeros((1, 1)))
    return SyncedFramePair(frame_index=index, rgb=rgb, nir=nir, skew_ns=skew_ns)


class TestRingBuffer(unittest.TestCase):
    def test_rejects_non_positive_capacity(self):
        with self.assertRaises(ValueError):
            RingBuffer(capacity=0)

    def test_default_capacity_matches_spec_temporal_window(self):
        buf = RingBuffer()
        self.assertEqual(buf.capacity, SPEC_TEMPORAL_WINDOW_FRAMES)
        self.assertEqual(SPEC_TEMPORAL_WINDOW_FRAMES, 120)

    def test_sustains_a_full_120_frame_window(self):
        buf = RingBuffer(capacity=120)

        for i in range(120):
            buf.push(_pair(i))

        self.assertTrue(buf.is_full)
        self.assertEqual(len(buf), 120)
        self.assertEqual(buf.pushed_frame_count, 120)
        self.assertEqual(buf.dropped_frame_count, 0)

    def test_window_keeps_rolling_past_capacity(self):
        buf = RingBuffer(capacity=10)

        for i in range(25):
            buf.push(_pair(i))

        window = buf.window()
        self.assertEqual(len(window), 10)
        self.assertEqual(window[0].frame_index, 15)
        self.assertEqual(window[-1].frame_index, 24)
        self.assertEqual(buf.pushed_frame_count, 25)

    def test_gap_in_frame_index_raises_but_still_appends_and_counts_the_drop(self):
        buf = RingBuffer(capacity=10)
        buf.push(_pair(0))

        with self.assertRaises(FrameGapError) as ctx:
            buf.push(_pair(4))  # frames 1,2,3 never arrived

        self.assertEqual(ctx.exception.gap, 3)
        self.assertEqual(buf.dropped_frame_count, 3)
        self.assertEqual(len(buf), 2)  # the gapped frame was still appended
        self.assertEqual(buf.window()[-1].frame_index, 4)

    def test_out_of_order_or_duplicate_index_raises_without_appending(self):
        buf = RingBuffer(capacity=10)
        buf.push(_pair(5))

        with self.assertRaises(FrameReorderError):
            buf.push(_pair(5))
        with self.assertRaises(FrameReorderError):
            buf.push(_pair(3))

        self.assertEqual(len(buf), 1)
        self.assertEqual(buf.dropped_frame_count, 0)


if __name__ == "__main__":
    unittest.main()
