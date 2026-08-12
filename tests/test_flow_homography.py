"""Tests for the pixel-to-metre site homography (spec 4.3, bead
ocean-vision-v1-bag)."""

import unittest

import numpy as np

from ov1.flow.homography import Homography, PointBehindCameraError


class TestHomography(unittest.TestCase):
    def test_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            Homography(matrix=np.eye(2))

    def test_rejects_non_finite_matrix(self):
        matrix = np.eye(3)
        matrix[0, 0] = np.nan
        with self.assertRaises(ValueError):
            Homography(matrix=matrix)

    def test_rejects_singular_matrix(self):
        with self.assertRaises(ValueError):
            Homography(matrix=np.zeros((3, 3)))

    def test_affine_scale_maps_pixels_to_metres(self):
        # A pure scale-and-translate homography (bottom row [0, 0, 1]) is a
        # special case with no perspective distortion -- easiest to check by
        # hand: (x, y) -> (0.05*x + 1.0, 0.05*y - 2.0).
        matrix = np.array([[0.05, 0, 1.0], [0, 0.05, -2.0], [0, 0, 1]], dtype=np.float64)
        homography = Homography(matrix=matrix)

        points_px = np.array([[0.0, 0.0], [100.0, 40.0]])
        metres = homography.pixel_to_metres(points_px)

        expected = np.array([[1.0, -2.0], [6.0, 0.0]])
        np.testing.assert_allclose(metres, expected)

    def test_perspective_homography_divides_by_w(self):
        # A genuine perspective term (nonzero bottom-left entries) requires
        # the homogeneous divide; check against a hand-computed point.
        matrix = np.array([[1.0, 0, 0], [0, 1.0, 0], [0.01, 0, 1.0]], dtype=np.float64)
        homography = Homography(matrix=matrix)

        metres = homography.pixel_to_metres(np.array([[50.0, 20.0]]))

        w = 0.01 * 50.0 + 1.0
        expected = np.array([[50.0 / w, 20.0 / w]])
        np.testing.assert_allclose(metres, expected)

    def test_raises_on_point_behind_camera(self):
        matrix = np.array([[1.0, 0, 0], [0, 1.0, 0], [-1.0, 0, 1.0]], dtype=np.float64)
        homography = Homography(matrix=matrix)

        with self.assertRaises(PointBehindCameraError):
            homography.pixel_to_metres(np.array([[10.0, 0.0]]))

    def test_rejects_wrong_point_shape(self):
        homography = Homography(matrix=np.eye(3))
        with self.assertRaises(ValueError):
            homography.pixel_to_metres(np.array([1.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
