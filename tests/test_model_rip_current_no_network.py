"""The rip current classifier sits on the inference -> alert path (spec 3,
4.3), which per spec 5.2 must never touch the network. This patches socket
construction to fail loudly (this repo's standard no-network pattern, e.g.
`test_flow_no_network.py`) and drives a full classify cycle through it.
"""

import socket
import unittest
from unittest import mock

import numpy as np

from ov1.flow import Homography
from ov1.flow.optical_flow import VelocityField
from ov1.model.rip_current import RipCurrentClassifier, ShorelineNormal


class TestRipCurrentClassifierDoesNotTouchTheNetwork(unittest.TestCase):
    def test_update_never_constructs_a_socket(self):
        classifier = RipCurrentClassifier(
            homography=Homography(matrix=np.eye(3, dtype=np.float64)),
            shoreline_normal=ShorelineNormal(vector=np.array([1.0, 0.0])),
            min_sustained_s=2.0 / 30.0,
        )
        field = VelocityField(
            positions_px=np.array([[10.0, 0.0], [10.0, 2.0]]),
            velocity_mps=np.array([[1.0, 0.0], [1.0, 0.0]]),
            dt_s=1.0 / 30.0,
        )

        def _forbidden_socket(*args, **kwargs):
            raise AssertionError("rip current classification attempted to open a network socket")

        with mock.patch.object(socket, "socket", side_effect=_forbidden_socket), \
                mock.patch.object(socket, "create_connection", side_effect=_forbidden_socket):
            observation = classifier.update(field)
            observation = classifier.update(field)

        self.assertTrue(observation.is_active)


if __name__ == "__main__":
    unittest.main()
