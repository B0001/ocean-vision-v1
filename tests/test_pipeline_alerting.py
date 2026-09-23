import unittest

import numpy as np

from ov1.model.reconstruction_trigger import DrowningObservation
from ov1.model.rip_current import RipChannel, RipCurrentObservation
from ov1.pipeline.alerting import DROWNING, RIP_CURRENT, AlertEngine, NodePosition

NODE = NodePosition(latitude_deg=21.28, longitude_deg=-157.83)  # test value, not a real site


def _drown(alerting):
    return DrowningObservation(score=1.0, above_threshold=alerting, sustained_s=0.0, is_alerting=alerting)


def _rip(active):
    channel = RipChannel(centroid_m=np.array([10.0, 2.0]), width_m=4.0, mean_speed_mps=1.0, point_count=3)
    return RipCurrentObservation(channels=(channel,) if active else (), sustained_s=0.0, is_active=active)


def _engine():
    calls = []
    engine = AlertEngine(
        relay=lambda a: calls.append(("relay", a)),
        gps_broadcast=lambda a: calls.append(("gps", a)),
        node_position=NODE,
    )
    return engine, calls


class TestFusion(unittest.TestCase):
    def test_either_head_alone_fires(self):
        engine, _ = _engine()
        self.assertEqual([a.kind for a in engine.update(_drown(True), _rip(False))], [DROWNING])
        engine, _ = _engine()
        alerts = engine.update(_drown(False), _rip(True))
        self.assertEqual([a.kind for a in alerts], [RIP_CURRENT])
        self.assertEqual(alerts[0].channel_centroids_m, ((10.0, 2.0),))
        self.assertEqual(alerts[0].node_position, NODE)

    def test_both_heads_same_step_drowning_first_relay_before_gps(self):
        engine, calls = _engine()
        engine.update(_drown(True), _rip(True))
        self.assertEqual(
            [(sink, a.kind) for sink, a in calls],
            [("relay", DROWNING), ("gps", DROWNING), ("relay", RIP_CURRENT), ("gps", RIP_CURRENT)],
        )

    def test_quiet_heads_fire_nothing(self):
        engine, calls = _engine()
        for _ in range(10):
            self.assertEqual(engine.update(_drown(False), _rip(False)), [])
        self.assertEqual(calls, [])


class TestDedupAndRearm(unittest.TestCase):
    def test_one_alert_per_onset_then_rearm(self):
        engine, calls = _engine()
        seq = [True, True, True, False, True, True]
        fired = [engine.update(_drown(s), None) for s in seq]
        self.assertEqual([len(f) for f in fired], [1, 0, 0, 0, 1, 0])
        self.assertEqual([a.step for f in fired for a in f], [0, 4])
        self.assertEqual(len(calls), 4)  # 2 onsets x (relay + gps)

    def test_none_observation_keeps_head_state(self):
        engine, _ = _engine()
        engine.update(_drown(True), None)
        # Head A not scored this step: must neither re-fire nor re-arm.
        self.assertEqual(engine.update(None, None), [])
        self.assertEqual(engine.update(_drown(True), None), [])

    def test_heads_latch_independently(self):
        engine, _ = _engine()
        engine.update(_drown(True), None)
        self.assertEqual([a.kind for a in engine.update(_drown(True), _rip(True))], [RIP_CURRENT])

    def test_failed_sink_does_not_swallow_the_alert(self):
        failures = [True]
        relayed = []

        def relay(alert):
            if failures.pop() if failures else False:
                raise OSError("relay not responding")
            relayed.append(alert)

        engine = AlertEngine(relay=relay, gps_broadcast=lambda a: None, node_position=NODE)
        with self.assertRaises(OSError):
            engine.update(_drown(True), None)
        self.assertEqual([a.step for a in engine.update(_drown(True), None)], [1])
        self.assertEqual(len(relayed), 1)


class TestDeterminism(unittest.TestCase):
    def test_same_observations_same_alerts(self):
        rng = np.random.default_rng(0)
        seq = [(bool(a), bool(b)) for a, b in rng.integers(0, 2, size=(500, 2))]

        def run():
            engine, calls = _engine()
            alerts = [a for d, r in seq for a in engine.update(_drown(d), _rip(r))]
            return alerts, calls

        first, second = run(), run()
        self.assertEqual(first, second)
        self.assertGreater(len(first[0]), 10)


class TestNodePosition(unittest.TestCase):
    def test_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            NodePosition(latitude_deg=91.0, longitude_deg=0.0)
        with self.assertRaises(ValueError):
            NodePosition(latitude_deg=0.0, longitude_deg=float("nan"))


if __name__ == "__main__":
    unittest.main()
