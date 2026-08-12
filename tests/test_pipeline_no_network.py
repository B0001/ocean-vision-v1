"""This glue sits squarely on the ingress -> inference path (spec 4.1, 4.2),
which per spec 5.2 must never touch the network. This patches socket
construction to fail loudly and drives a full RingBuffer-window ->
FrameChannelsPipeline -> InferenceTensorAssembler pass through it, so any
future change here that adds networking breaks this test.
"""

import socket
import unittest
from unittest import mock

import numpy as np

from ov1.ingress.frame import Frame
from ov1.ingress.ring_buffer import RingBuffer
from ov1.ingress.sync import SyncedFramePair
from ov1.pipeline.window_assembly import (
    FrameChannelsPipeline,
    SiteCalibrationConfig,
    assemble_tensor_from_window,
)
from ov1.tensor.assemble import InferenceTensorAssembler


def _synced_pair(frame_index, height=16, width=16):
    rng = np.random.default_rng(frame_index)
    rgb = Frame(
        sensor_id="rgb_optical",
        frame_index=frame_index,
        timestamp_ns=frame_index,
        data=rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8),
    )
    nir_data = rng.uniform(80.0, 120.0, size=(height, width, 4))
    nir = Frame(sensor_id="nir_850nm", frame_index=frame_index, timestamp_ns=frame_index, data=nir_data)
    return SyncedFramePair(frame_index=frame_index, rgb=rgb, nir=nir, skew_ns=0)


class TestPipelineDoesNotTouchTheNetwork(unittest.TestCase):
    def test_full_glue_never_constructs_a_socket(self):
        window_frames = 4
        config = SiteCalibrationConfig(clip_limit=2.0, tile_grid_size=(8, 8), dolp_threshold=0.5)

        def _forbidden_socket(*args, **kwargs):
            raise AssertionError("ingress-to-tensor glue attempted to open a network socket")

        with mock.patch.object(socket, "socket", side_effect=_forbidden_socket), \
                mock.patch.object(socket, "create_connection", side_effect=_forbidden_socket):
            pipeline = FrameChannelsPipeline.from_site_config(config)
            assembler = InferenceTensorAssembler(target_height=8, target_width=8, window_frames=window_frames)

            buf = RingBuffer(capacity=window_frames)
            for i in range(window_frames):
                buf.push(_synced_pair(i))

            result = assemble_tensor_from_window(buf.window(), pipeline, assembler, pin=False)

        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
