import unittest

import numpy as np

from ov1.ingress import SyntheticFrameSource


class TestSyntheticFrameSource(unittest.TestCase):
    def test_reads_frames_in_order_with_sequential_frame_index(self):
        frames = [(0, np.zeros((2, 2))), (1000, np.ones((2, 2)))]
        source = SyntheticFrameSource("rgb", frames)

        first = source.read()
        second = source.read()

        self.assertEqual(first.sensor_id, "rgb")
        self.assertEqual(first.frame_index, 0)
        self.assertEqual(first.timestamp_ns, 0)
        self.assertEqual(second.frame_index, 1)
        self.assertEqual(second.timestamp_ns, 1000)

    def test_returns_none_once_exhausted(self):
        source = SyntheticFrameSource("nir", [(0, np.zeros((1, 1)))])

        source.read()

        self.assertIsNone(source.read())

    def test_frame_rejects_negative_index_or_timestamp(self):
        from ov1.ingress import Frame

        with self.assertRaises(ValueError):
            Frame(sensor_id="rgb", frame_index=-1, timestamp_ns=0, data=np.zeros((1, 1)))
        with self.assertRaises(ValueError):
            Frame(sensor_id="rgb", frame_index=0, timestamp_ns=-1, data=np.zeros((1, 1)))


if __name__ == "__main__":
    unittest.main()
