"""Sparse optical flow sits on the ingress -> inference path (spec 3, 4.3),
which per spec 5.2 must never touch the network. This patches socket
construction to fail loudly and drives seeding + tracking through it, so any
future change here that adds networking breaks this test.
"""

import socket
import unittest
from unittest import mock

import numpy as np

from ov1.flow import Homography, SparseFlowTracker
from ov1.flow.optical_flow import _synthetic_drift_clip


class TestOpticalFlowDoesNotTouchTheNetwork(unittest.TestCase):
    def test_seed_and_step_never_construct_a_socket(self):
        homography = Homography(matrix=np.array([[0.05, 0, 0], [0, 0.05, 0], [0, 0, 1]], dtype=np.float64))
        tracker = SparseFlowTracker(homography=homography, frame_period_s=1.0 / 30.0)
        clip = _synthetic_drift_clip(n_frames=3)

        def _forbidden_socket(*args, **kwargs):
            raise AssertionError("optical flow tracking attempted to open a network socket")

        with mock.patch.object(socket, "socket", side_effect=_forbidden_socket), \
                mock.patch.object(socket, "create_connection", side_effect=_forbidden_socket):
            tracker.seed(clip[0])
            field = tracker.step(clip[1])
            field = tracker.step(clip[2])

        self.assertIsNotNone(field)


if __name__ == "__main__":
    unittest.main()
