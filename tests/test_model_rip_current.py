"""Tests for the spec 4.3 rip current classifier / Head B (bead
ocean-vision-v1-xe3). All velocity fields here are hand-built synthetic
tensors -- this validates the classifier's gating logic against known
inputs, not real foam-tracer or ocean-current behavior; there is no
annotated rip footage in this container to validate against (see the
module docstring in ov1.model.rip_current and this bead's handoff).
"""

import unittest

import numpy as np

from ov1.flow import Homography
from ov1.flow.optical_flow import VelocityField
from ov1.model.rip_current import (
    RipCurrentClassifier,
    ShorelineNormal,
    self_check,
)

IDENTITY_HOMOGRAPHY = Homography(matrix=np.eye(3, dtype=np.float64))
SEAWARD_X = ShorelineNormal(vector=np.array([1.0, 0.0]))
DT_S = 1.0 / 30.0


def _field(positions_px, velocity_mps, dt_s=DT_S):
    return VelocityField(
        positions_px=np.asarray(positions_px, dtype=np.float64),
        velocity_mps=np.asarray(velocity_mps, dtype=np.float64),
        dt_s=dt_s,
    )


def _classifier(**overrides):
    kwargs = dict(
        homography=IDENTITY_HOMOGRAPHY,
        shoreline_normal=SEAWARD_X,
        min_sustained_s=3 * DT_S,
    )
    kwargs.update(overrides)
    return RipCurrentClassifier(**kwargs)


class TestRipCurrentSelfCheck(unittest.TestCase):
    def test_self_check_passes(self):
        self_check()


class TestShorelineNormalValidation(unittest.TestCase):
    def test_rejects_non_unit_vector(self):
        with self.assertRaises(ValueError):
            ShorelineNormal(vector=np.array([2.0, 0.0]))

    def test_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            ShorelineNormal(vector=np.array([1.0, 0.0, 0.0]))

    def test_tangent_is_perpendicular(self):
        normal = ShorelineNormal(vector=np.array([1.0, 0.0]))
        self.assertAlmostEqual(float(np.dot(normal.vector, normal.tangent)), 0.0)


class TestRipCurrentClassifierConstruction(unittest.TestCase):
    def test_rejects_non_positive_min_sustained_s(self):
        with self.assertRaises(ValueError):
            _classifier(min_sustained_s=0.0)

    def test_rejects_non_positive_velocity_threshold(self):
        with self.assertRaises(ValueError):
            _classifier(velocity_threshold_mps=0.0)

    def test_rejects_non_positive_channel_width(self):
        with self.assertRaises(ValueError):
            _classifier(max_channel_width_m=0.0)

    def test_rejects_min_channel_points_below_two(self):
        with self.assertRaises(ValueError):
            _classifier(min_channel_points=1)


class TestDirectionGate(unittest.TestCase):
    def test_alongshore_flow_never_qualifies(self):
        classifier = _classifier()
        positions = [[10.0, 0.0], [10.0, 2.0]]
        velocity = [[0.0, 2.0], [0.0, 2.0]]  # fast, but purely alongshore
        observation = classifier.update(_field(positions, velocity))
        self.assertEqual(observation.channels, ())

    def test_shoreward_flow_never_qualifies(self):
        classifier = _classifier()
        positions = [[10.0, 0.0], [10.0, 2.0]]
        velocity = [[-2.0, 0.0], [-2.0, 0.0]]  # fast, but toward shore
        observation = classifier.update(_field(positions, velocity))
        self.assertEqual(observation.channels, ())

    def test_seaward_flow_qualifies_direction_gate(self):
        classifier = _classifier()
        positions = [[10.0, 0.0], [10.0, 2.0]]
        velocity = [[2.0, 0.0], [2.0, 0.0]]
        observation = classifier.update(_field(positions, velocity))
        self.assertEqual(len(observation.channels), 1)


class TestMagnitudeGate(unittest.TestCase):
    def test_below_threshold_speed_does_not_qualify(self):
        classifier = _classifier(velocity_threshold_mps=0.5)
        positions = [[10.0, 0.0], [10.0, 2.0]]
        velocity = [[0.4, 0.0], [0.4, 0.0]]
        observation = classifier.update(_field(positions, velocity))
        self.assertEqual(observation.channels, ())

    def test_above_threshold_speed_qualifies(self):
        classifier = _classifier(velocity_threshold_mps=0.5)
        positions = [[10.0, 0.0], [10.0, 2.0]]
        velocity = [[0.6, 0.0], [0.6, 0.0]]
        observation = classifier.update(_field(positions, velocity))
        self.assertEqual(len(observation.channels), 1)


class TestChannelWidthGate(unittest.TestCase):
    def test_narrow_group_forms_one_channel_under_width(self):
        classifier = _classifier(max_channel_width_m=15.0)
        positions = [[10.0, 0.0], [10.0, 5.0], [10.0, 9.0]]
        velocity = [[1.0, 0.0]] * 3
        observation = classifier.update(_field(positions, velocity))
        self.assertEqual(len(observation.channels), 1)
        channel = observation.channels[0]
        self.assertLess(channel.width_m, 15.0)
        self.assertEqual(channel.point_count, 3)

    def test_wide_spread_group_is_rejected(self):
        classifier = _classifier(max_channel_width_m=15.0, cluster_gap_m=100.0)
        positions = [[10.0, 0.0], [10.0, 20.0], [10.0, 40.0]]
        velocity = [[1.0, 0.0]] * 3
        observation = classifier.update(_field(positions, velocity))
        self.assertEqual(observation.channels, ())

    def test_two_disjoint_narrow_channels_both_reported(self):
        classifier = _classifier(max_channel_width_m=15.0, cluster_gap_m=15.0)
        positions = [
            [10.0, 0.0], [10.0, 5.0], [10.0, 9.0],   # channel A, width 9
            [10.0, 100.0], [10.0, 105.0], [10.0, 109.0],  # channel B, width 9, far away
        ]
        velocity = [[1.0, 0.0]] * 6
        observation = classifier.update(_field(positions, velocity))
        self.assertEqual(len(observation.channels), 2)
        for channel in observation.channels:
            self.assertLess(channel.width_m, 15.0)
            self.assertEqual(channel.point_count, 3)

    def test_single_point_cannot_form_a_channel(self):
        classifier = _classifier(min_channel_points=2)
        positions = [[10.0, 0.0]]
        velocity = [[1.0, 0.0]]
        observation = classifier.update(_field(positions, velocity))
        self.assertEqual(observation.channels, ())


class TestSustainGating(unittest.TestCase):
    def test_not_active_until_sustained_duration_met(self):
        min_sustained_s = 3 * DT_S
        classifier = _classifier(min_sustained_s=min_sustained_s)
        positions = [[10.0, 0.0], [10.0, 2.0]]
        velocity = [[1.0, 0.0], [1.0, 0.0]]
        field = _field(positions, velocity)

        observations = [classifier.update(field) for _ in range(5)]
        self.assertFalse(observations[0].is_active)
        self.assertFalse(observations[1].is_active)
        self.assertTrue(observations[2].is_active)
        self.assertTrue(observations[4].is_active)

    def test_a_gap_step_resets_sustain_timer(self):
        min_sustained_s = 3 * DT_S
        classifier = _classifier(min_sustained_s=min_sustained_s)
        rip_field = _field([[10.0, 0.0], [10.0, 2.0]], [[1.0, 0.0], [1.0, 0.0]])
        calm_field = _field([[10.0, 0.0], [10.0, 2.0]], [[0.0, 0.0], [0.0, 0.0]])

        classifier.update(rip_field)
        classifier.update(rip_field)
        gap_observation = classifier.update(calm_field)
        self.assertEqual(gap_observation.sustained_s, 0.0)

        resumed = [classifier.update(rip_field) for _ in range(3)]
        self.assertFalse(resumed[0].is_active)
        self.assertTrue(resumed[2].is_active)

    def test_reset_clears_accumulated_sustain(self):
        classifier = _classifier(min_sustained_s=3 * DT_S)
        rip_field = _field([[10.0, 0.0], [10.0, 2.0]], [[1.0, 0.0], [1.0, 0.0]])
        classifier.update(rip_field)
        classifier.update(rip_field)
        self.assertGreater(classifier.sustained_s, 0.0)

        classifier.reset()
        self.assertEqual(classifier.sustained_s, 0.0)

    def test_empty_field_never_qualifies_and_resets(self):
        classifier = _classifier(min_sustained_s=3 * DT_S)
        rip_field = _field([[10.0, 0.0], [10.0, 2.0]], [[1.0, 0.0], [1.0, 0.0]])
        empty_field = _field(np.empty((0, 2)), np.empty((0, 2)))

        classifier.update(rip_field)
        observation = classifier.update(empty_field)
        self.assertEqual(observation.channels, ())
        self.assertEqual(observation.sustained_s, 0.0)
        self.assertFalse(observation.is_active)


class TestFailsLoud(unittest.TestCase):
    def test_rejects_non_finite_velocity(self):
        classifier = _classifier()
        positions = [[10.0, 0.0]]
        velocity = [[np.nan, 0.0]]
        with self.assertRaises(ValueError):
            classifier.update(_field(positions, velocity))

    def test_rejects_non_positive_dt(self):
        classifier = _classifier()
        with self.assertRaises(ValueError):
            classifier.update(_field([[10.0, 0.0]], [[1.0, 0.0]], dt_s=0.0))


if __name__ == "__main__":
    unittest.main()
