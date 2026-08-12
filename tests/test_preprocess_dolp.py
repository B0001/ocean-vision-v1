"""Tests for DoLP specular glare masking (spec 4.1, bead
ocean-vision-v1-bze). Runs against small synthetic polarization planes --
this validates the code path, not real sun-glint statistics; see the module
docstring in ov1.preprocess.dolp for that caveat.
"""

import unittest

import numpy as np

from ov1.preprocess import (
    DolpComputationError,
    SpecularGlareMasker,
    compute_dolp,
    compute_stokes,
)
from ov1.preprocess.dolp import _synthetic_polarized_frame, self_check


class TestComputeStokes(unittest.TestCase):
    def test_unpolarized_light_yields_zero_s1_s2(self):
        i0 = i45 = i90 = i135 = np.full((8, 8), 100.0)
        stokes = compute_stokes(i0, i45, i90, i135)

        np.testing.assert_array_equal(stokes.s1, np.zeros((8, 8)))
        np.testing.assert_array_equal(stokes.s2, np.zeros((8, 8)))
        np.testing.assert_array_equal(stokes.s0, np.full((8, 8), 200.0))

    def test_rejects_shape_mismatch(self):
        i0 = np.full((8, 8), 1.0)
        i45 = np.full((8, 8), 1.0)
        i90 = np.full((8, 8), 1.0)
        i135 = np.full((4, 4), 1.0)
        with self.assertRaises(ValueError):
            compute_stokes(i0, i45, i90, i135)

    def test_rejects_non_2d_plane(self):
        i0 = np.full((8, 8, 1), 1.0)
        i45 = i90 = i135 = np.full((8, 8), 1.0)
        with self.assertRaises(ValueError):
            compute_stokes(i0, i45, i90, i135)

    def test_rejects_negative_intensity(self):
        i0 = np.full((8, 8), -1.0)
        i45 = i90 = i135 = np.full((8, 8), 1.0)
        with self.assertRaises(ValueError):
            compute_stokes(i0, i45, i90, i135)

    def test_rejects_non_finite_input(self):
        i0 = np.full((8, 8), 1.0)
        i0[0, 0] = np.nan
        i45 = i90 = i135 = np.full((8, 8), 1.0)
        with self.assertRaises(DolpComputationError):
            compute_stokes(i0, i45, i90, i135)


class TestComputeDolp(unittest.TestCase):
    def test_self_check_passes(self):
        # Runs the same baseline/glare/mask assertions as
        # `uv run python -m ov1.preprocess.dolp` standalone.
        self_check()

    def test_dolp_in_unit_range_and_frame_resolution(self):
        i0, i45, i90, i135, _ = _synthetic_polarized_frame(height=32, width=48)
        stokes = compute_stokes(i0, i45, i90, i135)

        dolp = compute_dolp(stokes)

        self.assertEqual(dolp.shape, (32, 48))
        self.assertTrue(np.all(dolp >= 0.0))
        self.assertTrue(np.all(dolp <= 1.0))

    def test_fully_polarized_light_yields_dolp_one(self):
        i0 = np.full((4, 4), 100.0)
        i90 = np.full((4, 4), 0.0)
        i45 = np.full((4, 4), 50.0)
        i135 = np.full((4, 4), 50.0)
        stokes = compute_stokes(i0, i45, i90, i135)

        dolp = compute_dolp(stokes)

        np.testing.assert_allclose(dolp, 1.0)

    def test_zero_intensity_raises_rather_than_dividing_by_zero(self):
        i0 = i45 = i90 = i135 = np.zeros((4, 4))
        stokes = compute_stokes(i0, i45, i90, i135)

        with self.assertRaises(DolpComputationError):
            compute_dolp(stokes)

    def test_rejects_mismatched_stokes_shapes(self):
        from ov1.preprocess.dolp import StokesParameters

        stokes = StokesParameters(
            s0=np.full((4, 4), 1.0), s1=np.full((4, 4), 0.0), s2=np.full((2, 2), 0.0)
        )
        with self.assertRaises(ValueError):
            compute_dolp(stokes)


class TestSpecularGlareMasker(unittest.TestCase):
    def test_rejects_non_positive_threshold(self):
        with self.assertRaises(ValueError):
            SpecularGlareMasker(dolp_threshold=0.0)

    def test_rejects_threshold_above_one(self):
        with self.assertRaises(ValueError):
            SpecularGlareMasker(dolp_threshold=1.5)

    def test_mask_flags_high_dolp_pixels(self):
        dolp = np.array([[0.1, 0.9], [0.4, 0.6]])
        masker = SpecularGlareMasker(dolp_threshold=0.5)

        mask = masker.mask(dolp)

        expected = np.array([[False, True], [False, True]])
        np.testing.assert_array_equal(mask.glare, expected)
        np.testing.assert_array_equal(mask.suppression, np.where(expected, 0.0, 1.0))

    def test_mask_rejects_out_of_range_dolp(self):
        masker = SpecularGlareMasker(dolp_threshold=0.5)
        with self.assertRaises(ValueError):
            masker.mask(np.array([[1.5]]))

    def test_suppress_zeroes_only_glare_pixels_and_does_not_mutate_input(self):
        dolp = np.array([[0.1, 0.9]])
        plane = np.array([[50.0, 200.0]])
        original = plane.copy()
        masker = SpecularGlareMasker(dolp_threshold=0.5)

        suppressed = masker.suppress(plane, dolp)

        np.testing.assert_array_equal(suppressed, np.array([[50.0, 0.0]]))
        np.testing.assert_array_equal(plane, original)

    def test_suppress_rejects_shape_mismatch(self):
        masker = SpecularGlareMasker(dolp_threshold=0.5)
        with self.assertRaises(ValueError):
            masker.suppress(np.zeros((2, 2)), np.zeros((3, 3)))

    def test_suppress_preserves_uint8_dtype(self):
        dolp = np.array([[0.1, 0.9]])
        plane = np.array([[50, 200]], dtype=np.uint8)
        masker = SpecularGlareMasker(dolp_threshold=0.5)

        suppressed = masker.suppress(plane, dolp)

        self.assertEqual(suppressed.dtype, np.uint8)
        np.testing.assert_array_equal(suppressed, np.array([[50, 0]], dtype=np.uint8))


class TestFrameResolutionSmoke(unittest.TestCase):
    def test_runs_on_spec_frame_resolution(self):
        # Spec 4.2's (H, W) = (1080, 1920). Not a timing measurement -- this
        # container has no Jetson to time against the 250ms/frame budget --
        # only a smoke test that the vectorized numpy path completes on the
        # spec'd resolution without pathological behavior.
        height, width = 1080, 1920
        rng = np.random.default_rng(42)
        i0 = rng.uniform(50, 150, size=(height, width))
        i45 = rng.uniform(50, 150, size=(height, width))
        i90 = rng.uniform(50, 150, size=(height, width))
        i135 = rng.uniform(50, 150, size=(height, width))

        stokes = compute_stokes(i0, i45, i90, i135)
        dolp = compute_dolp(stokes)
        masker = SpecularGlareMasker(dolp_threshold=0.8)
        mask = masker.mask(dolp)

        self.assertEqual(dolp.shape, (height, width))
        self.assertEqual(mask.glare.shape, (height, width))


if __name__ == "__main__":
    unittest.main()
