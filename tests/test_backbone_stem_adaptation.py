"""Tests for the 4-channel patch-embedding stem adaptation (bead
ocean-vision-v1-dxj, spec 3's ViT embedding branch).

Runs against a tiny, randomly-initialized `VideoMAEModel` built from a small
custom config -- this validates the adaptation *logic* (weight preservation,
new-channel init, shape wiring) fast and without touching the vendored
checkpoint, per this repo's "small synthetic tensors" testing convention.
Real-checkpoint loading is `test_backbone_no_network.py`'s job.
"""

import unittest

import torch
from transformers import VideoMAEConfig, VideoMAEModel

from ov1.model.backbone import StemAdaptationError, _adapt_patch_embedding_channels


def _tiny_model(num_channels: int = 3) -> VideoMAEModel:
    config = VideoMAEConfig(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        num_frames=4,
        image_size=32,
        patch_size=16,
        tubelet_size=2,
        num_channels=num_channels,
        qkv_bias=True,
    )
    return VideoMAEModel(config)


class TestStemAdaptation(unittest.TestCase):
    def test_grows_conv_in_channels(self):
        model = _tiny_model(num_channels=3)
        _adapt_patch_embedding_channels(model, num_channels=4)

        conv = model.embeddings.patch_embeddings.projection
        self.assertEqual(conv.in_channels, 4)
        self.assertEqual(model.embeddings.patch_embeddings.num_channels, 4)
        self.assertEqual(model.config.num_channels, 4)

    def test_preserves_pretrained_rgb_weights(self):
        model = _tiny_model(num_channels=3)
        original_weight = model.embeddings.patch_embeddings.projection.weight.detach().clone()

        _adapt_patch_embedding_channels(model, num_channels=4)

        new_weight = model.embeddings.patch_embeddings.projection.weight
        torch.testing.assert_close(new_weight[:, :3], original_weight)

    def test_new_channel_initialized_as_mean_of_original(self):
        model = _tiny_model(num_channels=3)
        original_weight = model.embeddings.patch_embeddings.projection.weight.detach().clone()
        expected_new_channel = original_weight.mean(dim=1)

        _adapt_patch_embedding_channels(model, num_channels=4)

        new_weight = model.embeddings.patch_embeddings.projection.weight
        torch.testing.assert_close(new_weight[:, 3], expected_new_channel)

    def test_preserves_bias(self):
        model = _tiny_model(num_channels=3)
        original_bias = model.embeddings.patch_embeddings.projection.bias.detach().clone()

        _adapt_patch_embedding_channels(model, num_channels=4)

        torch.testing.assert_close(model.embeddings.patch_embeddings.projection.bias, original_bias)

    def test_adapted_model_forwards_on_4channel_input(self):
        model = _tiny_model(num_channels=3)
        _adapt_patch_embedding_channels(model, num_channels=4)
        model.eval()

        x = torch.randn(1, 4, 4, 32, 32)  # (B, T, C, H, W) per VideoMAEModel's own convention
        with torch.no_grad():
            out = model(x)

        expected_seq_len = (4 // 2) * (32 // 16) * (32 // 16)  # tubelet=2, patch=16
        self.assertEqual(tuple(out.last_hidden_state.shape), (1, expected_seq_len, 8))
        self.assertTrue(torch.isfinite(out.last_hidden_state).all())

    def test_rejects_shrinking_the_stem(self):
        model = _tiny_model(num_channels=3)
        with self.assertRaises(StemAdaptationError):
            _adapt_patch_embedding_channels(model, num_channels=2)

    def test_rejects_unchanged_channel_count(self):
        model = _tiny_model(num_channels=3)
        with self.assertRaises(StemAdaptationError):
            _adapt_patch_embedding_channels(model, num_channels=3)


if __name__ == "__main__":
    unittest.main()
