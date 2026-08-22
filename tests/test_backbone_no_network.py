"""Acceptance test for bead ocean-vision-v1-dxj: "Model loads offline with
the network disabled." Patches socket construction to fail loudly (this
repo's standard no-network pattern, e.g. `test_pipeline_no_network.py`) and
then loads the real vendored checkpoint through `VideoTransformerBackbone`,
so any future change that makes the loader reach the network breaks this
test instead of silently degrading spec 5.2's offline requirement.

Loads the real ~376MB `models/videomae-base/model.safetensors` from disk
(~4s on this container's CPU, measured) rather than a tiny synthetic config
-- this repo's "small synthetic tensors" testing convention (see
`test_backbone_stem_adaptation.py`) covers the adaptation *logic*, but only
a real checkpoint load can demonstrate the acceptance criterion. No forward
pass here: `VideoTransformerBackbone.embed()` on real 16x224x224 input is
several more seconds of CPU time on top of the load, exercised interactively
per `backbone.self_check`'s docstring, not in the automated suite.
"""

import socket
import unittest
from unittest import mock

from ov1.model.backbone import (
    DEFAULT_MODEL_DIR,
    TARGET_NUM_CHANNELS,
    EmbeddingShape,
    VideoTransformerBackbone,
)


@unittest.skipUnless(
    (DEFAULT_MODEL_DIR / "model.safetensors").is_file(),
    f"vendored checkpoint not found at {DEFAULT_MODEL_DIR} -- run scripts/vendor_backbone.py first",
)
class TestBackboneLoadsOfflineWithNetworkDisabled(unittest.TestCase):
    def test_loads_with_network_sockets_blocked(self):
        def _forbidden_socket(*args, **kwargs):
            raise AssertionError("backbone load attempted to open a network socket")

        with mock.patch.object(socket, "socket", side_effect=_forbidden_socket), \
                mock.patch.object(socket, "create_connection", side_effect=_forbidden_socket):
            backbone = VideoTransformerBackbone()

        self.assertEqual(backbone.num_channels, TARGET_NUM_CHANNELS)
        self.assertEqual(backbone.original_num_channels, 3)

    def test_documented_embedding_shape_matches_checkpoint_geometry(self):
        # Spec 4.2's window is (1,4,120,1080,1920): 120 frames at 1080x1920.
        # Sanity-checked here against the checkpoint's actual
        # (patch_size, tubelet_size, hidden_size), not asserted as a
        # performance or correctness claim about the ocean footage itself.
        backbone = VideoTransformerBackbone()
        shape = backbone.embedding_shape(num_frames=16, height=224, width=224)
        self.assertEqual(shape, EmbeddingShape(seq_len=1568, hidden_size=768))
        self.assertEqual(shape.as_tensor_shape(batch_size=1), (1, 1568, 768))


if __name__ == "__main__":
    unittest.main()
