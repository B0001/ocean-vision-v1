"""Tensor assembly sits on the ingress -> inference path (spec 4.2), which
per spec 5.2 must never touch the network. This patches socket construction
to fail loudly and drives assembly (including the pin-memory attempt)
through it, so any future change here that adds networking breaks this test.
"""

import socket
import unittest
from unittest import mock

from ov1.tensor.assemble import InferenceTensorAssembler, _synthetic_frame_channels


class TestTensorAssemblyDoesNotTouchTheNetwork(unittest.TestCase):
    def test_assemble_never_constructs_a_socket(self):
        window_frames, height, width = 4, 16, 16
        frames = [_synthetic_frame_channels(32, 32, seed=i) for i in range(window_frames)]
        assembler = InferenceTensorAssembler(
            target_height=height, target_width=width, window_frames=window_frames
        )

        def _forbidden_socket(*args, **kwargs):
            raise AssertionError("tensor assembly attempted to open a network socket")

        with mock.patch.object(socket, "socket", side_effect=_forbidden_socket), \
                mock.patch.object(socket, "create_connection", side_effect=_forbidden_socket):
            result = assembler.assemble(frames)

        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
