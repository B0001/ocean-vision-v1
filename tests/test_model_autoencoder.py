"""Tests for the Head A spatiotemporal autoencoder (bead ocean-vision-v1-133,
spec 4.2). Runs entirely against small synthetic tensors and synthetic
"baseline"-labeled clips -- this validates the reconstruction-shape
invariant and the training loop's mechanics, not anything about real ocean
footage or reconstruction quality on real distress dynamics. See
`ov1.model.autoencoder`'s own module docstring for that caveat, and
`ov1.tensor.assemble`'s for why small synthetic planes are this repo's
standard test fixture.
"""

import socket
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch.utils.data import DataLoader

from ov1.data.dataset import (
    BaselineWindowDataset,
    ClipFrameSource,
    InMemoryClipFrameSource,
    WindowBatch,
    collate_window_samples,
)
from ov1.data.manifest import ClipProvenance
from ov1.model.autoencoder import (
    AutoencoderShapeError,
    LatencyMeasurement,
    SpatiotemporalAutoencoder,
    TrainingResult,
    measure_inference_latency,
    self_check,
    train_autoencoder,
)
from ov1.tensor.assemble import SPEC_CHANNELS, FrameChannels


def _clip_provenance(clip_id, split="train", **overrides):
    base = dict(
        clip_id=clip_id,
        path=Path(f"{clip_id}.npz"),
        label="baseline",
        source="synthetic-baseline-v1",
        site_id="site-A",
        capture_date="2026-01-15",
        reviewed_by="alice",
        split=split,
    )
    base.update(overrides)
    return ClipProvenance(**base)


def _synthetic_frames(n_frames, height, width, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(n_frames):
        frames.append(
            FrameChannels(
                l=rng.integers(0, 256, size=(height, width), dtype=np.uint8),
                a=rng.integers(0, 256, size=(height, width), dtype=np.uint8),
                b=rng.integers(0, 256, size=(height, width), dtype=np.uint8),
                dolp=rng.random((height, width)),
            )
        )
    return frames


class _FrameStore:
    def __init__(self, store):
        self._store = store

    def __call__(self, clip: ClipProvenance) -> ClipFrameSource:
        return InMemoryClipFrameSource(self._store[clip.clip_id])


class TestForwardShapeInvariant(unittest.TestCase):
    def test_forward_returns_tensor_matching_input_shape(self):
        model = SpatiotemporalAutoencoder(base_channels=4, depth=2, norm_groups=2)
        for batch_size, t, height, width in [(1, 4, 8, 8), (2, 4, 16, 16), (1, 8, 16, 24)]:
            with self.subTest(batch_size=batch_size, t=t, height=height, width=width):
                x = torch.rand(batch_size, SPEC_CHANNELS, t, height, width)
                reconstruction = model(x)
                self.assertEqual(tuple(reconstruction.shape), tuple(x.shape))
                self.assertEqual(reconstruction.dtype, x.dtype)

    def test_forward_at_default_depth_matches_spec_resolution_factors(self):
        # Spec 4.2's 1080x1920 divides evenly by 2**DEFAULT_DEPTH (=8); this
        # exercises that exact claim at a tractably small multiple of 8
        # rather than the full 1080x1920 (too slow/large for this container
        # per the repo's synthetic-tensor testing convention).
        model = SpatiotemporalAutoencoder()
        x = torch.rand(1, SPEC_CHANNELS, 4, 24, 40)  # 24, 40 divisible by 8
        reconstruction = model(x)
        self.assertEqual(tuple(reconstruction.shape), tuple(x.shape))

    def test_forward_rejects_non_divisible_height_width(self):
        model = SpatiotemporalAutoencoder(base_channels=4, depth=2, norm_groups=2)  # factor = 4
        x = torch.rand(1, SPEC_CHANNELS, 4, 15, 16)  # 15 not divisible by 4
        with self.assertRaises(AutoencoderShapeError):
            model(x)

    def test_forward_rejects_wrong_channel_count(self):
        model = SpatiotemporalAutoencoder(base_channels=4, depth=2, norm_groups=2)
        x = torch.rand(1, SPEC_CHANNELS + 1, 4, 16, 16)
        with self.assertRaises(ValueError):
            model(x)

    def test_forward_rejects_wrong_ndim(self):
        model = SpatiotemporalAutoencoder(base_channels=4, depth=2, norm_groups=2)
        x = torch.rand(SPEC_CHANNELS, 4, 16, 16)  # missing batch dim
        with self.assertRaises(ValueError):
            model(x)

    def test_temporal_stride_two_requires_divisible_temporal_length(self):
        model = SpatiotemporalAutoencoder(base_channels=2, depth=2, norm_groups=2, temporal_stride=2)
        ok = torch.rand(1, SPEC_CHANNELS, 4, 16, 16)  # 4 divisible by temporal_downsample_factor=4
        self.assertEqual(tuple(model(ok).shape), tuple(ok.shape))
        bad = torch.rand(1, SPEC_CHANNELS, 3, 16, 16)  # 3 not divisible by 4
        with self.assertRaises(AutoencoderShapeError):
            model(bad)


class TestConstructorValidation(unittest.TestCase):
    def test_rejects_non_positive_depth(self):
        with self.assertRaises(ValueError):
            SpatiotemporalAutoencoder(depth=0)

    def test_rejects_invalid_temporal_stride(self):
        with self.assertRaises(ValueError):
            SpatiotemporalAutoencoder(temporal_stride=3)

    def test_rejects_channel_count_not_divisible_by_norm_groups(self):
        # base_channels=3, depth=1 -> single block with 3 channels; default
        # norm_groups=8 does not divide 3.
        with self.assertRaises(ValueError):
            SpatiotemporalAutoencoder(base_channels=3, depth=1, norm_groups=8)


class TestTrainingLoopMechanics(unittest.TestCase):
    def test_loss_decreases_on_synthetic_baseline_windows(self):
        # "Baseline" here means gate_baseline_clips-eligible provenance
        # (label="baseline", no exclusion_reason, split="train") over
        # synthetic random frames -- this proves the training loop's
        # mechanics converge on its own gated-data path, not that the model
        # learns anything about real aquatic motion.
        torch.manual_seed(0)
        window_frames, height, width = 4, 16, 16
        clips = [_clip_provenance(f"c{i}") for i in range(3)]
        store = {clip.clip_id: _synthetic_frames(window_frames, height, width, seed=i) for i, clip in enumerate(clips)}
        dataset = BaselineWindowDataset(
            clips, _FrameStore(store), window_frames=window_frames, target_height=height, target_width=width
        )
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_window_samples, shuffle=True)

        model = SpatiotemporalAutoencoder(base_channels=4, depth=2, norm_groups=2)
        result = train_autoencoder(model, loader, epochs=10, lr=1e-2)

        self.assertIsInstance(result, TrainingResult)
        self.assertEqual(len(result.epoch_losses), 10)
        self.assertLess(result.final_loss, result.initial_loss)

    def test_zero_epochs_raises(self):
        model = SpatiotemporalAutoencoder(base_channels=2, depth=1, norm_groups=2)
        batches = [WindowBatch(tensor=torch.rand(1, SPEC_CHANNELS, 2, 4, 4), clip_ids=["a"], start_frames=[0])]
        with self.assertRaises(ValueError):
            train_autoencoder(model, batches, epochs=0)

    def test_empty_batches_raises(self):
        model = SpatiotemporalAutoencoder(base_channels=2, depth=1, norm_groups=2)
        with self.assertRaises(ValueError):
            train_autoencoder(model, [], epochs=1)


class TestLatencyMeasurementUtility(unittest.TestCase):
    def test_returns_positive_timings_for_requested_iterations(self):
        model = SpatiotemporalAutoencoder(base_channels=2, depth=1, norm_groups=2)
        x = torch.rand(1, SPEC_CHANNELS, 2, 4, 4)
        result = measure_inference_latency(model, x, num_iterations=3, warmup_iterations=1)
        self.assertIsInstance(result, LatencyMeasurement)
        self.assertEqual(result.num_iterations, 3)
        self.assertGreater(result.mean_ms, 0.0)
        self.assertLessEqual(result.min_ms, result.mean_ms)
        self.assertGreaterEqual(result.max_ms, result.mean_ms)
        self.assertEqual(result.device, "cpu")
        # Deliberately not asserted against spec 2's <250ms target: this is
        # a CPU dev-container measurement, not a Jetson AGX Orin one (spec
        # 5.1) -- see LatencyMeasurement's docstring.

    def test_rejects_non_positive_iterations(self):
        model = SpatiotemporalAutoencoder(base_channels=2, depth=1, norm_groups=2)
        x = torch.rand(1, SPEC_CHANNELS, 2, 4, 4)
        with self.assertRaises(ValueError):
            measure_inference_latency(model, x, num_iterations=0)


class TestSelfCheck(unittest.TestCase):
    def test_self_check_runs_without_raising(self):
        result = self_check()
        self.assertIsInstance(result, TrainingResult)


class TestDoesNotTouchTheNetwork(unittest.TestCase):
    def test_forward_and_train_never_construct_a_socket(self):
        # Head A's forward pass sits on the ingress -> inference -> alert
        # path (spec 4.2), which per spec 5.2 must never touch the network.
        # This model has no pretrained checkpoint to load (unlike
        # ov1.model.backbone) -- weights are randomly initialized -- so this
        # mainly guards against a future change (e.g. a logging/telemetry
        # call) accidentally introducing one on this path.
        model = SpatiotemporalAutoencoder(base_channels=2, depth=1, norm_groups=2)
        x = torch.rand(2, SPEC_CHANNELS, 2, 4, 4)
        batches = [WindowBatch(tensor=x, clip_ids=["a", "b"], start_frames=[0, 0])]

        def _forbidden_socket(*args, **kwargs):
            raise AssertionError("autoencoder forward/train attempted to open a network socket")

        with mock.patch.object(socket, "socket", side_effect=_forbidden_socket), \
                mock.patch.object(socket, "create_connection", side_effect=_forbidden_socket):
            model(x)
            train_autoencoder(model, batches, epochs=1)


if __name__ == "__main__":
    unittest.main()
