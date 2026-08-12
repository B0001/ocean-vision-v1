"""The ingress path must never open a socket (spec 5.2: offline operation is
a hard requirement on the ingress -> inference -> alert path). This patches
socket construction to fail loudly and drives the full ingress pipeline
through it, so any future change that adds networking here breaks this test.
"""

import socket
import unittest
from unittest import mock

import numpy as np

from ov1.ingress import DualSensorSynchronizer, RingBuffer, SensorStarvedError, SyntheticFrameSource

FRAME_RATE_HZ = 60.0
PERIOD_NS = round(1e9 / FRAME_RATE_HZ)


def _px(i):
    return np.full((4, 4), i % 256, dtype=np.uint8)


class TestIngressDoesNotTouchTheNetwork(unittest.TestCase):
    def test_full_pipeline_never_constructs_a_socket(self):
        n = 130
        rgb_frames = [(i * PERIOD_NS, _px(i)) for i in range(n)]
        nir_frames = [(i * PERIOD_NS, _px(i)) for i in range(n)]

        def _forbidden_socket(*args, **kwargs):
            raise AssertionError("ingress pipeline attempted to open a network socket")

        with mock.patch.object(socket, "socket", side_effect=_forbidden_socket), \
                mock.patch.object(socket, "create_connection", side_effect=_forbidden_socket):
            rgb = SyntheticFrameSource("rgb_optical", rgb_frames)
            nir = SyntheticFrameSource("nir_850nm", nir_frames)
            sync = DualSensorSynchronizer(rgb, nir, nominal_frame_rate_hz=FRAME_RATE_HZ)
            buf = RingBuffer(capacity=120)

            while not buf.is_full:
                try:
                    pair = sync.next_pair()
                except SensorStarvedError:
                    break
                buf.push(pair)

        self.assertTrue(buf.is_full)


if __name__ == "__main__":
    unittest.main()
