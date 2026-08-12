"""Tests for CIE Lab depth-attenuation normalization (spec 4.1, bead
ocean-vision-v1-3qj). Runs against small synthetic frames -- this validates
the code path, not real ocean turbidity; see the module docstring in
ov1.preprocess.lab for that caveat.
"""

import unittest

import cv2
import numpy as np

from ov1.preprocess import LabNormalizer
from ov1.preprocess.lab import _synthetic_turbid_frame, self_check


class TestLabNormalizer(unittest.TestCase):
    def test_self_check_passes(self):
        # Runs the same channel-selective-mutation + histogram-de-skew
        # assertions as `uv run python -m ov1.preprocess.lab` standalone.
        self_check()

    def test_emits_l_a_b_planes_matching_frame_shape(self):
        frame = _synthetic_turbid_frame(height=32, width=48)
        normalizer = LabNormalizer(clip_limit=2.0, tile_grid_size=(8, 8))

        result = normalizer.normalize(frame)

        for plane in (result.l, result.a, result.b):
            self.assertEqual(plane.shape, (32, 48))
            self.assertEqual(plane.dtype, np.uint8)

    def test_l_bit_identical_to_unstretched_conversion(self):
        frame = _synthetic_turbid_frame()
        normalizer = LabNormalizer(clip_limit=3.5, tile_grid_size=(4, 4))

        result = normalizer.normalize(frame)
        expected_l = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)[:, :, 0]

        np.testing.assert_array_equal(result.l, expected_l)

    def test_ab_histograms_de_skewed_on_turbid_footage(self):
        frame = _synthetic_turbid_frame()
        normalizer = LabNormalizer(clip_limit=2.0, tile_grid_size=(8, 8))

        result = normalizer.normalize(frame)
        unstretched = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        unstretched_a, unstretched_b = unstretched[:, :, 1], unstretched[:, :, 2]

        self.assertGreater(float(result.a.std()), float(unstretched_a.std()))
        self.assertGreater(float(result.b.std()), float(unstretched_b.std()))

    def test_mutation_is_channel_selective(self):
        frame = _synthetic_turbid_frame()
        normalizer = LabNormalizer(clip_limit=2.0, tile_grid_size=(8, 8))

        result = normalizer.normalize(frame)
        unstretched = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)

        np.testing.assert_array_equal(result.l, unstretched[:, :, 0])
        self.assertFalse(np.array_equal(result.a, unstretched[:, :, 1]))
        self.assertFalse(np.array_equal(result.b, unstretched[:, :, 2]))

    def test_rejects_wrong_shape(self):
        normalizer = LabNormalizer(clip_limit=2.0, tile_grid_size=(8, 8))
        with self.assertRaises(ValueError):
            normalizer.normalize(np.zeros((8, 8), dtype=np.uint8))

    def test_rejects_wrong_dtype(self):
        normalizer = LabNormalizer(clip_limit=2.0, tile_grid_size=(8, 8))
        with self.assertRaises(ValueError):
            normalizer.normalize(np.zeros((8, 8, 3), dtype=np.float32))

    def test_rejects_non_positive_clip_limit(self):
        with self.assertRaises(ValueError):
            LabNormalizer(clip_limit=0.0, tile_grid_size=(8, 8))

    def test_rejects_non_positive_tile_grid(self):
        with self.assertRaises(ValueError):
            LabNormalizer(clip_limit=2.0, tile_grid_size=(0, 8))


if __name__ == "__main__":
    unittest.main()
