"""Tests for the spec 4.2 reconstruction-error score and sliding-window
continuity trigger (bead ocean-vision-v1-wtg). All score traces here are
hand-built synthetic numbers -- this validates the continuity state
machine against known inputs, not real drowning dynamics; there is no
ocean footage or trained autoencoder in this container to validate
against (see the module docstring in ov1.model.reconstruction_trigger).
"""

import unittest

import torch

from ov1.model.reconstruction_trigger import (
    DrowningTrigger,
    ReconstructionScoreError,
    evaluate_synthetic_traces,
    reconstruction_error_score,
    self_check,
)

TAU = 1.0
DT_S = 1.0 / 30.0
MIN_CONTINUOUS_S = 3.0


def _trigger(tau=TAU, min_continuous_s=MIN_CONTINUOUS_S):
    return DrowningTrigger(tau_drowning=tau, min_continuous_s=min_continuous_s)


class ReconstructionErrorScoreTest(unittest.TestCase):
    def test_is_sum_of_squares_not_mean(self):
        x = torch.zeros(1, 1, 2, 2, 2)
        x_hat = torch.full((1, 1, 2, 2, 2), 2.0)
        # 8 elements, residual 2 each -> sum of squares = 32 (a mean would be 4).
        self.assertEqual(reconstruction_error_score(x, x_hat), 32.0)

    def test_perfect_reconstruction_scores_zero(self):
        x = torch.randn(1, 2, 3, 4, 4)
        self.assertEqual(reconstruction_error_score(x, x.clone()), 0.0)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ReconstructionScoreError):
            reconstruction_error_score(torch.zeros(1, 1, 2, 2, 2), torch.zeros(1, 1, 2, 2, 3))

    def test_non_finite_raises(self):
        x = torch.zeros(2, 2)
        for bad in (float("nan"), float("inf")):
            with self.subTest(bad=bad):
                x_hat = torch.full((2, 2), bad)
                with self.assertRaises(ReconstructionScoreError):
                    reconstruction_error_score(x, x_hat)
                with self.assertRaises(ReconstructionScoreError):
                    reconstruction_error_score(x_hat, x)


class DrowningTriggerTest(unittest.TestCase):
    def test_rejects_non_positive_calibration(self):
        for kwargs in ({"tau_drowning": 0.0}, {"tau_drowning": -1.0}, {"min_continuous_s": 0.0}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    DrowningTrigger(
                        **{"tau_drowning": TAU, "min_continuous_s": MIN_CONTINUOUS_S, **kwargs}
                    )

    def test_spike_shorter_than_window_does_not_fire(self):
        trigger = _trigger()
        for _ in range(int(2.9 / DT_S)):
            self.assertFalse(trigger.update(TAU + 5.0, DT_S).is_alerting)
        self.assertLess(trigger.sustained_s, MIN_CONTINUOUS_S)

    def test_spike_past_window_fires(self):
        trigger = _trigger()
        fired = any(trigger.update(TAU + 5.0, DT_S).is_alerting for _ in range(int(4.0 / DT_S)))
        self.assertTrue(fired)

    def test_sub_threshold_reading_resets_continuity(self):
        trigger = _trigger()
        for _ in range(int(2.5 / DT_S)):
            trigger.update(TAU + 5.0, DT_S)
        observation = trigger.update(TAU - 0.5, DT_S)
        self.assertEqual(observation.sustained_s, 0.0)
        self.assertFalse(observation.above_threshold)
        # Clock restarts from zero rather than resuming.
        last = None
        for _ in range(int(2.5 / DT_S)):
            last = trigger.update(TAU + 5.0, DT_S)
        self.assertFalse(last.is_alerting)

    def test_score_exactly_at_tau_is_not_above(self):
        # Spec 4.2 is S_r > tau, strictly.
        self.assertFalse(_trigger().update(TAU, DT_S).above_threshold)

    def test_reset_clears_continuity(self):
        trigger = _trigger()
        for _ in range(int(2.5 / DT_S)):
            trigger.update(TAU + 5.0, DT_S)
        trigger.reset()
        self.assertEqual(trigger.sustained_s, 0.0)

    def test_update_rejects_bad_inputs(self):
        trigger = _trigger()
        with self.assertRaises(ValueError):
            trigger.update(float("nan"), DT_S)
        with self.assertRaises(ValueError):
            trigger.update(TAU + 5.0, 0.0)
        with self.assertRaises(ValueError):
            trigger.update(TAU + 5.0, -DT_S)


class SyntheticTraceTest(unittest.TestCase):
    def test_empty_trace_lists_raise(self):
        trace = [[TAU + 5.0] * 200]
        for pos, neg in (([], trace), (trace, [])):
            with self.subTest(pos=bool(pos)):
                with self.assertRaises(ValueError):
                    evaluate_synthetic_traces(
                        pos, neg, tau_drowning=TAU, min_continuous_s=MIN_CONTINUOUS_S, dt_s=DT_S
                    )

    def test_counts_intended_misses(self):
        # One "positive" trace too short to fire, one "negative" trace that does.
        result = evaluate_synthetic_traces(
            [[TAU + 5.0] * int(1.0 / DT_S)],
            [[TAU + 5.0] * int(5.0 / DT_S)],
            tau_drowning=TAU,
            min_continuous_s=MIN_CONTINUOUS_S,
            dt_s=DT_S,
        )
        self.assertEqual(result.false_negatives, 1)
        self.assertEqual(result.false_positives, 1)
        self.assertEqual(result.synthetic_fnr, 1.0)
        self.assertEqual(result.synthetic_fpr, 1.0)

    def test_self_check(self):
        result = self_check()
        self.assertEqual(result.false_negatives, 0)
        self.assertEqual(result.false_positives, 0)


if __name__ == "__main__":
    unittest.main()
