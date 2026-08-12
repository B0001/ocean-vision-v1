"""Tests for the sparse Lucas-Kanade velocity field tracker (spec 3, 4.3,
bead ocean-vision-v1-bag). Runs against small synthetic clips -- this
validates the tracker's mechanics, not real foam-tracer or ocean-current
behavior; see the module docstring in ov1.flow.optical_flow for that
caveat.
"""

import unittest
from unittest import mock

import numpy as np

from ov1.flow import (
    FlowFieldError,
    Homography,
    NoTrackablePointsError,
    SparseFlowTracker,
)
from ov1.flow.optical_flow import _synthetic_drift_clip, self_check

IDENTITY_SCALE_HOMOGRAPHY = Homography(
    matrix=np.array([[0.05, 0, 0], [0, 0.05, 0], [0, 0, 1]], dtype=np.float64)
)


class TestSparseFlowTrackerSelfCheck(unittest.TestCase):
    def test_self_check_passes(self):
        self_check()


class TestSparseFlowTrackerDriftDirection(unittest.TestCase):
    def test_known_drift_recovered_in_metres(self):
        dx_px_per_frame = 5.0
        frame_period_s = 1.0 / 30.0
        metres_per_pixel = 0.05

        tracker = SparseFlowTracker(homography=IDENTITY_SCALE_HOMOGRAPHY, frame_period_s=frame_period_s)
        clip = _synthetic_drift_clip(n_frames=4, dx_px_per_frame=dx_px_per_frame, dy_px_per_frame=0.0)
        tracker.seed(clip[0])

        expected_vx = metres_per_pixel * dx_px_per_frame / frame_period_s
        for frame in clip[1:]:
            field = tracker.step(frame)
            mean_v = field.velocity_mps.mean(axis=0)
            self.assertGreater(mean_v[0], 0.0)
            self.assertLess(abs(mean_v[0] - expected_vx), 0.15 * expected_vx)
            self.assertLess(abs(mean_v[1]), 0.1 * expected_vx)

    def test_reversed_drift_direction_is_negative(self):
        frame_period_s = 1.0 / 30.0
        tracker = SparseFlowTracker(homography=IDENTITY_SCALE_HOMOGRAPHY, frame_period_s=frame_period_s)
        clip = _synthetic_drift_clip(n_frames=4, dx_px_per_frame=-5.0, dy_px_per_frame=0.0)
        tracker.seed(clip[0])

        for frame in clip[1:]:
            field = tracker.step(frame)
            self.assertLess(field.velocity_mps.mean(axis=0)[0], 0.0)

    def test_dt_s_recorded_on_field(self):
        frame_period_s = 1.0 / 24.0
        tracker = SparseFlowTracker(homography=IDENTITY_SCALE_HOMOGRAPHY, frame_period_s=frame_period_s)
        clip = _synthetic_drift_clip(n_frames=2)
        tracker.seed(clip[0])

        field = tracker.step(clip[1])
        self.assertEqual(field.dt_s, frame_period_s)


class TestSparseFlowTrackerReseeding(unittest.TestCase):
    def test_low_tracked_count_triggers_reseed_back_toward_capacity(self):
        tracker = SparseFlowTracker(
            homography=IDENTITY_SCALE_HOMOGRAPHY,
            frame_period_s=1.0 / 30.0,
            max_corners=200,
            re_seed_fraction=0.9,
        )
        clip = _synthetic_drift_clip(n_frames=3, dx_px_per_frame=30.0)
        tracker.seed(clip[0])
        self.assertEqual(tracker.tracked_point_count, 200)

        tracker.step(clip[1])
        count_after_first_step = tracker.tracked_point_count

        tracker.step(clip[2])

        self.assertGreater(tracker.last_re_seed_count, 0, "second step should have re-seeded fresh points")
        self.assertGreater(
            tracker.tracked_point_count,
            count_after_first_step,
            "re-seeding should recover point count, not just track the shrinking survivor set",
        )

    def test_reseeded_points_respect_min_distance_from_survivors(self):
        tracker = SparseFlowTracker(
            homography=IDENTITY_SCALE_HOMOGRAPHY,
            frame_period_s=1.0 / 30.0,
            max_corners=200,
            min_distance=7.0,
            re_seed_fraction=0.9,
        )
        clip = _synthetic_drift_clip(n_frames=3, dx_px_per_frame=30.0)
        tracker.seed(clip[0])
        field1 = tracker.step(clip[1])
        field2 = tracker.step(clip[2])

        if tracker.last_re_seed_count == 0:
            self.skipTest("this step did not re-seed; nothing to check")

        survivors = field2.positions_px
        # step() appends fresh re-seeded points after the survivors in its
        # internal state; step() itself only returns the tracked (survivor)
        # positions, so checking the fresh points' spacing requires reading
        # that internal state directly.
        all_points = tracker._prev_points  # noqa: SLF001 -- whitebox check of internal invariant
        fresh_points = all_points[-tracker.last_re_seed_count:]
        for fx, fy in fresh_points:
            dists = np.linalg.norm(survivors - np.array([fx, fy]), axis=1)
            self.assertGreaterEqual(dists.min(), 7.0 - 1.0)  # +/-1px rounding slack


class TestSparseFlowTrackerFailureModes(unittest.TestCase):
    def test_seed_rejects_blank_frame(self):
        tracker = SparseFlowTracker(homography=IDENTITY_SCALE_HOMOGRAPHY, frame_period_s=1.0 / 30.0)
        blank = np.zeros((64, 64), dtype=np.uint8)
        with self.assertRaises(NoTrackablePointsError):
            tracker.seed(blank)

    def test_step_before_seed_raises(self):
        tracker = SparseFlowTracker(homography=IDENTITY_SCALE_HOMOGRAPHY, frame_period_s=1.0 / 30.0)
        clip = _synthetic_drift_clip(n_frames=1)
        with self.assertRaises(RuntimeError):
            tracker.step(clip[0])

    def test_rejects_wrong_ndim_frame(self):
        tracker = SparseFlowTracker(homography=IDENTITY_SCALE_HOMOGRAPHY, frame_period_s=1.0 / 30.0)
        with self.assertRaises(ValueError):
            tracker.seed(np.zeros((64, 64, 3), dtype=np.uint8))

    def test_rejects_wrong_dtype_frame(self):
        tracker = SparseFlowTracker(homography=IDENTITY_SCALE_HOMOGRAPHY, frame_period_s=1.0 / 30.0)
        with self.assertRaises(ValueError):
            tracker.seed(np.zeros((64, 64), dtype=np.float32))

    def test_rejects_frame_shape_change_mid_track(self):
        tracker = SparseFlowTracker(homography=IDENTITY_SCALE_HOMOGRAPHY, frame_period_s=1.0 / 30.0)
        clip = _synthetic_drift_clip(n_frames=2, height=64, width=64)
        tracker.seed(clip[0])
        with self.assertRaises(ValueError):
            tracker.step(np.zeros((32, 32), dtype=np.uint8))

    def test_all_points_lost_raises_flow_field_error(self):
        tracker = SparseFlowTracker(homography=IDENTITY_SCALE_HOMOGRAPHY, frame_period_s=1.0 / 30.0)
        clip = _synthetic_drift_clip(n_frames=2)
        tracker.seed(clip[0])

        n_points = tracker.tracked_point_count
        fake_status = np.zeros((n_points, 1), dtype=np.uint8)
        fake_next = tracker._prev_points.reshape(-1, 1, 2).copy()  # noqa: SLF001
        with mock.patch("cv2.calcOpticalFlowPyrLK", return_value=(fake_next, fake_status, None)):
            with self.assertRaises(FlowFieldError):
                tracker.step(clip[1])

    def test_non_finite_flow_output_raises_flow_field_error(self):
        tracker = SparseFlowTracker(homography=IDENTITY_SCALE_HOMOGRAPHY, frame_period_s=1.0 / 30.0)
        clip = _synthetic_drift_clip(n_frames=2)
        tracker.seed(clip[0])

        n_points = tracker.tracked_point_count
        fake_status = np.ones((n_points, 1), dtype=np.uint8)
        fake_next = tracker._prev_points.reshape(-1, 1, 2).copy()  # noqa: SLF001
        fake_next[0, 0, 0] = np.nan
        with mock.patch("cv2.calcOpticalFlowPyrLK", return_value=(fake_next, fake_status, None)):
            with self.assertRaises(FlowFieldError):
                tracker.step(clip[1])


class TestVelocityFieldValidation(unittest.TestCase):
    def test_rejects_mismatched_shapes(self):
        from ov1.flow.optical_flow import VelocityField

        with self.assertRaises(ValueError):
            VelocityField(
                positions_px=np.zeros((3, 2)),
                velocity_mps=np.zeros((2, 2)),
                dt_s=1.0 / 30.0,
            )


if __name__ == "__main__":
    unittest.main()
