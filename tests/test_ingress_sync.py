import unittest

import numpy as np

from ov1.ingress import (
    DualSensorSynchronizer,
    SensorStarvedError,
    SensorSyncError,
    SyntheticFrameSource,
)

FRAME_RATE_HZ = 60.0
PERIOD_NS = round(1e9 / FRAME_RATE_HZ)


def _px(i):
    return np.full((2, 2), i, dtype=np.uint8)


class TestDualSensorSynchronizer(unittest.TestCase):
    def test_rejects_non_positive_frame_rate_or_skew(self):
        rgb = SyntheticFrameSource("rgb", [])
        nir = SyntheticFrameSource("nir", [])
        with self.assertRaises(ValueError):
            DualSensorSynchronizer(rgb, nir, nominal_frame_rate_hz=0)
        with self.assertRaises(ValueError):
            DualSensorSynchronizer(rgb, nir, nominal_frame_rate_hz=60, max_skew_frames=0)

    def test_pairs_frames_within_tolerance(self):
        rgb = SyntheticFrameSource("rgb", [(i * PERIOD_NS, _px(i)) for i in range(5)])
        # Sub-frame jitter well inside a 1-frame-period tolerance.
        nir = SyntheticFrameSource("nir", [(i * PERIOD_NS + 1000, _px(i)) for i in range(5)])
        sync = DualSensorSynchronizer(rgb, nir, nominal_frame_rate_hz=FRAME_RATE_HZ)

        pairs = [sync.next_pair() for _ in range(5)]

        self.assertEqual(sync.dropped_frame_count, 0)
        for i, pair in enumerate(pairs):
            self.assertEqual(pair.frame_index, i)
            self.assertLess(abs(pair.skew_ns), sync.skew_tolerance_ns)

    def test_skew_beyond_tolerance_raises_and_counts_a_drop(self):
        rgb = SyntheticFrameSource("rgb", [(i * PERIOD_NS, _px(i)) for i in range(3)])
        # NIR frame at index 1 arrives a full extra frame period late.
        nir_timestamps = [0, 1 * PERIOD_NS + 2 * PERIOD_NS, 2 * PERIOD_NS + 2 * PERIOD_NS]
        nir = SyntheticFrameSource("nir", [(t, _px(i)) for i, t in enumerate(nir_timestamps)])
        sync = DualSensorSynchronizer(rgb, nir, nominal_frame_rate_hz=FRAME_RATE_HZ)

        sync.next_pair()  # frame 0 aligns fine
        with self.assertRaises(SensorSyncError) as ctx:
            sync.next_pair()

        self.assertEqual(sync.dropped_frame_count, 1)
        self.assertEqual(ctx.exception.lagging_sensor, "rgb")

    def test_resyncs_after_a_skew_error_by_advancing_the_lagging_sensor(self):
        rgb = SyntheticFrameSource("rgb", [(i * PERIOD_NS, _px(i)) for i in range(4)])
        # NIR is missing frame index 1 entirely, shifting everything after it
        # one full frame period ahead relative to RGB.
        nir_frames = [(0, _px(0))] + [((i + 1) * PERIOD_NS, _px(i)) for i in range(1, 4)]
        nir = SyntheticFrameSource("nir", nir_frames)
        sync = DualSensorSynchronizer(rgb, nir, nominal_frame_rate_hz=FRAME_RATE_HZ)

        sync.next_pair()  # index 0 aligns
        with self.assertRaises(SensorSyncError):
            sync.next_pair()  # rgb index 1 vs nir index 2's timestamp -> drop rgb 1

        pair = sync.next_pair()  # retry: rgb index 2 now compared against pending nir

        self.assertEqual(sync.dropped_frame_count, 1)
        self.assertLess(abs(pair.skew_ns), sync.skew_tolerance_ns)

    def test_starved_source_raises(self):
        rgb = SyntheticFrameSource("rgb", [(0, _px(0))])
        nir = SyntheticFrameSource("nir", [])
        sync = DualSensorSynchronizer(rgb, nir, nominal_frame_rate_hz=FRAME_RATE_HZ)

        with self.assertRaises(SensorStarvedError) as ctx:
            sync.next_pair()

        self.assertEqual(ctx.exception.starved_sensor, "nir")


if __name__ == "__main__":
    unittest.main()
