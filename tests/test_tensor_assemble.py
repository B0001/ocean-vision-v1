"""Tests for spec-4.2 inference tensor assembly (bead ocean-vision-v1-3fy).
Runs against small synthetic planes at a small target resolution -- this
validates the assembly code path (shape, dtype, channel order, resizing,
pinning), not real Lab/DoLP data; see the module docstring in
ov1.tensor.assemble for that caveat.
"""

import unittest

import cv2
import numpy as np
import torch

from ov1.preprocess.dolp import SpecularGlareMasker
from ov1.preprocess.lab import LabFrame
from ov1.tensor import (
    CHANNEL_A,
    CHANNEL_B,
    CHANNEL_DOLP,
    CHANNEL_L,
    CHANNEL_ORDER,
    SPEC_CHANNELS,
    SPEC_HEIGHT,
    SPEC_TEMPORAL_WINDOW_FRAMES,
    SPEC_WIDTH,
    AssembledTensor,
    FrameChannels,
    InferenceTensorAssembler,
    TensorAssemblyError,
)
from ov1.tensor.assemble import _synthetic_frame_channels, self_check


def _frame(height=8, width=12, fill=1.0):
    l_plane = np.full((height, width), fill, dtype=np.uint8)
    a_plane = np.full((height, width), fill, dtype=np.uint8)
    b_plane = np.full((height, width), fill, dtype=np.uint8)
    dolp_plane = np.full((height, width), 0.5)
    return FrameChannels(l=l_plane, a=a_plane, b=b_plane, dolp=dolp_plane)


class TestSpecConstants(unittest.TestCase):
    def test_spec_shape_constants_match_spec_4_2(self):
        self.assertEqual(SPEC_CHANNELS, 4)
        self.assertEqual(SPEC_TEMPORAL_WINDOW_FRAMES, 120)
        self.assertEqual(SPEC_HEIGHT, 1080)
        self.assertEqual(SPEC_WIDTH, 1920)

    def test_channel_order_is_fixed_and_documented(self):
        self.assertEqual(CHANNEL_ORDER, ("L", "a*", "b*", "DoLP"))
        self.assertEqual((CHANNEL_L, CHANNEL_A, CHANNEL_B, CHANNEL_DOLP), (0, 1, 2, 3))

    def test_assembler_defaults_to_spec_shape(self):
        assembler = InferenceTensorAssembler()
        self.assertEqual(
            assembler.output_shape, (1, 4, SPEC_TEMPORAL_WINDOW_FRAMES, SPEC_HEIGHT, SPEC_WIDTH)
        )


class TestAssembleShapeAndDtype(unittest.TestCase):
    def test_self_check_passes(self):
        self_check()

    def test_assembled_tensor_matches_configured_shape_and_dtype(self):
        window_frames, height, width = 5, 16, 24
        frames = [_synthetic_frame_channels(32, 40, seed=i) for i in range(window_frames)]
        assembler = InferenceTensorAssembler(
            target_height=height, target_width=width, window_frames=window_frames
        )

        result = assembler.assemble(frames)

        self.assertIsInstance(result, AssembledTensor)
        self.assertEqual(tuple(result.tensor.shape), (1, SPEC_CHANNELS, window_frames, height, width))
        self.assertEqual(result.tensor.dtype, torch.float32)

    def test_channel_order_matches_source_planes(self):
        height, width = 8, 8
        frame = _frame(height, width, fill=1)
        # Distinct constant values per channel so a mis-ordered stack shows up.
        frame = FrameChannels(
            l=np.full((height, width), 10, dtype=np.uint8),
            a=np.full((height, width), 20, dtype=np.uint8),
            b=np.full((height, width), 30, dtype=np.uint8),
            dolp=np.full((height, width), 0.75),
        )
        assembler = InferenceTensorAssembler(target_height=height, target_width=width, window_frames=1)

        result = assembler.assemble([frame])

        # L/a*/b* are rescaled onto DoLP's native [0, 1] range (bead
        # ocean-vision-v1-3hw): channel value / 255, not the raw uint8 value.
        tensor = result.tensor[0]  # drop batch dim -> (C,T,H,W)
        self.assertTrue(torch.allclose(tensor[CHANNEL_L, 0], torch.full((height, width), 10.0 / 255.0)))
        self.assertTrue(torch.allclose(tensor[CHANNEL_A, 0], torch.full((height, width), 20.0 / 255.0)))
        self.assertTrue(torch.allclose(tensor[CHANNEL_B, 0], torch.full((height, width), 30.0 / 255.0)))
        self.assertTrue(torch.allclose(tensor[CHANNEL_DOLP, 0], torch.full((height, width), 0.75)))

    def test_l_a_b_channels_are_rescaled_onto_dolp_unit_range(self):
        # bead ocean-vision-v1-3hw's decision: uint8 [0, 255] L/a*/b* divide
        # onto DoLP's native [0, 1] float range rather than stacking raw
        # magnitudes up to 255x apart.
        height, width = 4, 4
        frame = FrameChannels(
            l=np.full((height, width), 255, dtype=np.uint8),
            a=np.full((height, width), 0, dtype=np.uint8),
            b=np.full((height, width), 128, dtype=np.uint8),
            dolp=np.full((height, width), 1.0),
        )
        assembler = InferenceTensorAssembler(target_height=height, target_width=width, window_frames=1)

        result = assembler.assemble([frame])

        tensor = result.tensor[0]
        self.assertTrue(torch.all(tensor >= 0.0) and torch.all(tensor <= 1.0))
        self.assertTrue(torch.allclose(tensor[CHANNEL_L, 0], torch.full((height, width), 1.0)))
        self.assertTrue(torch.allclose(tensor[CHANNEL_A, 0], torch.full((height, width), 0.0)))
        self.assertTrue(
            torch.allclose(tensor[CHANNEL_B, 0], torch.full((height, width), 128.0 / 255.0))
        )

    def test_downscale_matches_direct_cv2_resize(self):
        rng = np.random.default_rng(7)
        native_h, native_w = 64, 96
        target_h, target_w = 16, 24
        l_plane = rng.integers(0, 256, size=(native_h, native_w), dtype=np.uint8)
        frame = FrameChannels(
            l=l_plane,
            a=np.zeros((native_h, native_w), dtype=np.uint8),
            b=np.zeros((native_h, native_w), dtype=np.uint8),
            dolp=np.zeros((native_h, native_w)),
        )
        assembler = InferenceTensorAssembler(target_height=target_h, target_width=target_w, window_frames=1)

        result = assembler.assemble([frame])

        expected = cv2.resize(
            l_plane.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_AREA
        )
        # L is rescaled onto [0, 1] after resize (bead ocean-vision-v1-3hw).
        np.testing.assert_allclose(result.tensor[0, CHANNEL_L, 0].numpy(), expected / 255.0)


class TestRealResolutionSmoke(unittest.TestCase):
    def test_downscales_real_4k_frames_to_spec_1080p_shape(self):
        # A handful of frames at real 4K (2160, 3840), the "downscale 4K
        # ingress to 1080p" path spec 4.2 actually describes -- kept to a
        # few frames (not the full 120-frame window) so it stays a fast
        # smoke test, not a timing measurement. This container has no
        # Jetson to time the real 250ms/frame budget against.
        native_h, native_w = 2160, 3840
        window_frames = 3
        frames = [_synthetic_frame_channels(native_h, native_w, seed=i) for i in range(window_frames)]
        assembler = InferenceTensorAssembler(window_frames=window_frames)

        result = assembler.assemble(frames)

        self.assertEqual(
            tuple(result.tensor.shape), (1, SPEC_CHANNELS, window_frames, SPEC_HEIGHT, SPEC_WIDTH)
        )
        self.assertEqual(result.tensor.dtype, torch.float32)


class TestAssembleValidation(unittest.TestCase):
    def test_rejects_wrong_frame_count(self):
        assembler = InferenceTensorAssembler(target_height=8, target_width=8, window_frames=5)
        frames = [_frame() for _ in range(4)]
        with self.assertRaises(TensorAssemblyError):
            assembler.assemble(frames)

    def test_rejects_mismatched_plane_shapes_within_a_frame(self):
        assembler = InferenceTensorAssembler(target_height=8, target_width=8, window_frames=1)
        bad_frame = FrameChannels(
            l=np.zeros((8, 8), dtype=np.uint8),
            a=np.zeros((8, 8), dtype=np.uint8),
            b=np.zeros((4, 4), dtype=np.uint8),
            dolp=np.zeros((8, 8)),
        )
        with self.assertRaises(TensorAssemblyError):
            assembler.assemble([bad_frame])

    def test_rejects_non_2d_plane(self):
        assembler = InferenceTensorAssembler(target_height=8, target_width=8, window_frames=1)
        bad_frame = FrameChannels(
            l=np.zeros((8, 8, 1), dtype=np.uint8),
            a=np.zeros((8, 8), dtype=np.uint8),
            b=np.zeros((8, 8), dtype=np.uint8),
            dolp=np.zeros((8, 8)),
        )
        with self.assertRaises(TensorAssemblyError):
            assembler.assemble([bad_frame])

    def test_rejects_non_finite_values(self):
        assembler = InferenceTensorAssembler(target_height=8, target_width=8, window_frames=1)
        dolp = np.zeros((8, 8))
        dolp[0, 0] = np.nan
        bad_frame = FrameChannels(
            l=np.zeros((8, 8), dtype=np.uint8),
            a=np.zeros((8, 8), dtype=np.uint8),
            b=np.zeros((8, 8), dtype=np.uint8),
            dolp=dolp,
        )
        with self.assertRaises(TensorAssemblyError):
            assembler.assemble([bad_frame])

    def test_rejects_non_positive_target_dims(self):
        with self.assertRaises(ValueError):
            InferenceTensorAssembler(target_height=0, target_width=8)
        with self.assertRaises(ValueError):
            InferenceTensorAssembler(target_height=8, target_width=-1)

    def test_rejects_non_positive_window(self):
        with self.assertRaises(ValueError):
            InferenceTensorAssembler(window_frames=0)


class TestPinnedMemoryPath(unittest.TestCase):
    def test_pin_true_reports_honestly_for_this_runtime(self):
        assembler = InferenceTensorAssembler(target_height=8, target_width=8, window_frames=1)
        result = assembler.assemble([_frame(8, 8)], pin=True)

        self.assertEqual(result.pinned, result.tensor.is_pinned())
        if torch.cuda.is_available():
            self.assertTrue(result.pinned)
            self.assertIsNone(result.pin_skipped_reason)
        else:
            # This container has no accelerator -- pin_memory() cannot
            # silently succeed. Confirmed on this hardware, not the Jetson
            # target (see handoff).
            self.assertFalse(result.pinned)
            self.assertTrue(result.pin_skipped_reason)

    def test_pin_false_skips_the_pinned_path_explicitly(self):
        assembler = InferenceTensorAssembler(target_height=8, target_width=8, window_frames=1)
        result = assembler.assemble([_frame(8, 8)], pin=False)

        self.assertFalse(result.pinned)
        self.assertFalse(result.tensor.is_pinned())
        self.assertIsNotNone(result.pin_skipped_reason)


class TestFrameChannelsFromLabAndDolp(unittest.TestCase):
    def test_without_masker_l_plane_passes_through_unmodified(self):
        lab = LabFrame(
            l=np.full((4, 4), 200, dtype=np.uint8),
            a=np.full((4, 4), 10, dtype=np.uint8),
            b=np.full((4, 4), 20, dtype=np.uint8),
        )
        dolp = np.full((4, 4), 0.9)

        channels = FrameChannels.from_lab_and_dolp(lab, dolp)

        np.testing.assert_array_equal(channels.l, lab.l)
        np.testing.assert_array_equal(channels.a, lab.a)
        np.testing.assert_array_equal(channels.b, lab.b)
        np.testing.assert_array_equal(channels.dolp, dolp)

    def test_with_masker_suppresses_l_plane_only_on_glare_pixels(self):
        lab = LabFrame(
            l=np.full((2, 2), 200, dtype=np.uint8),
            a=np.full((2, 2), 10, dtype=np.uint8),
            b=np.full((2, 2), 20, dtype=np.uint8),
        )
        dolp = np.array([[0.1, 0.9], [0.2, 0.95]])
        masker = SpecularGlareMasker(dolp_threshold=0.5)

        channels = FrameChannels.from_lab_and_dolp(lab, dolp, glare_masker=masker)

        expected_l = np.array([[200, 0], [200, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(channels.l, expected_l)
        # a*/b*/DoLP must be untouched by masking.
        np.testing.assert_array_equal(channels.a, lab.a)
        np.testing.assert_array_equal(channels.b, lab.b)
        np.testing.assert_array_equal(channels.dolp, dolp)


if __name__ == "__main__":
    unittest.main()
