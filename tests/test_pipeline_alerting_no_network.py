"""Frame ingress -> relay, end to end, with networking disabled (spec 5.2;
bead ocean-vision-v1-1ou). Patches socket construction to fail loudly (this
repo's standard no-network pattern, e.g. `test_pipeline_no_network.py`) and
drives synthetic sources -> synchronizer -> ring buffer -> Lab/DoLP ->
tensor -> autoencoder -> S_r -> DrowningTrigger (Head A), plus L-plane
optical flow -> RipCurrentClassifier (Head B), -> AlertEngine -> relay.

Latency caveat: measured on whatever machine runs this suite (a dev Mac, not
the Jetson AGX Orin edge node), at a toy 64x64 resolution with an untrained
autoencoder -- not spec 4.2's 1080x1920. It shows the alert path adds no
blocking dependency and fits 250 ms at this size; it is not evidence for
spec 2's latency target on the edge node at full resolution.
"""

import socket
import time
import unittest
from unittest import mock

import numpy as np
import torch

from ov1.flow import Homography
from ov1.flow.optical_flow import SparseFlowTracker
from ov1.ingress import DualSensorSynchronizer, RingBuffer, SyntheticFrameSource
from ov1.model.autoencoder import SpatiotemporalAutoencoder
from ov1.model.reconstruction_trigger import DrowningTrigger, reconstruction_error_score
from ov1.model.rip_current import RipCurrentClassifier, ShorelineNormal
from ov1.pipeline import (
    AlertEngine,
    FrameChannelsPipeline,
    NodePosition,
    SiteCalibrationConfig,
    assemble_tensor_from_window,
)
from ov1.tensor.assemble import InferenceTensorAssembler

FPS = 30.0
DT_S = 1.0 / FPS
PERIOD_NS = round(1e9 / FPS)
WINDOW = 120
SIDE = 64
LATENCY_BUDGET_MS = 250.0  # spec 2, "Inference Latency" target boundary


def _sources(n):
    rng = np.random.default_rng(0)
    base_rgb = rng.integers(0, 256, size=(SIDE, SIDE, 3), dtype=np.uint8)
    base_nir = rng.uniform(80.0, 120.0, size=(SIDE, SIDE, 4))
    # Rolling one textured image 1px/frame gives the LK tracker something to follow.
    rgb = [(i * PERIOD_NS, np.roll(base_rgb, i, axis=1)) for i in range(n)]
    nir = [(i * PERIOD_NS, np.roll(base_nir, i, axis=1)) for i in range(n)]
    return SyntheticFrameSource("rgb_optical", rgb), SyntheticFrameSource("nir_850nm", nir)


def _run_frame_to_relay():
    """Returns (alerts, latency_ns) for the one frame that completes Head A's
    3.0s continuity window. Everything before that frame is untimed warm-up."""
    torch.manual_seed(0)
    relay_times = []
    relayed = []

    def relay(alert):
        relay_times.append(time.perf_counter_ns())
        relayed.append(alert)

    engine = AlertEngine(
        relay=relay,
        gps_broadcast=lambda alert: None,
        node_position=NodePosition(latitude_deg=21.28, longitude_deg=-157.83),  # test value
    )
    config = SiteCalibrationConfig(clip_limit=2.0, tile_grid_size=(8, 8), dolp_threshold=0.5)  # test values
    pipeline = FrameChannelsPipeline.from_site_config(config)
    assembler = InferenceTensorAssembler(target_height=SIDE, target_width=SIDE, window_frames=WINDOW)
    model = SpatiotemporalAutoencoder().eval()
    # tau is a test value far below any untrained-model S_r, so the real
    # score qualifies; min_continuous_s is spec 4.2's 3.0s.
    trigger = DrowningTrigger(tau_drowning=1e-6, min_continuous_s=3.0)
    homography = Homography(matrix=np.eye(3, dtype=np.float64))
    tracker = SparseFlowTracker(homography=homography, frame_period_s=DT_S)
    classifier = RipCurrentClassifier(
        homography=homography,
        shoreline_normal=ShorelineNormal(vector=np.array([1.0, 0.0])),
        min_sustained_s=1.0,  # test value
    )

    rgb, nir = _sources(WINDOW + 1)
    sync = DualSensorSynchronizer(rgb, nir, nominal_frame_rate_hz=FPS)
    buf = RingBuffer(capacity=WINDOW)
    for _ in range(WINDOW):
        buf.push(sync.next_pair())
    tracker.seed(pipeline.convert(buf.window()[-1]).l)
    with torch.no_grad():  # boot-time warm-up, as the edge node would do once
        model(torch.zeros(assembler.output_shape))
    # Stand in for the preceding 3.0s - one step of above-tau windows.
    while trigger.sustained_s + DT_S < trigger.min_continuous_s:
        trigger.update(1.0, DT_S)

    t0 = time.perf_counter_ns()
    pair = sync.next_pair()  # frame ingress
    buf.push(pair)
    x = assemble_tensor_from_window(buf.window(), pipeline, assembler, pin=False).tensor
    with torch.no_grad():
        x_hat = model(x)
    drowning = trigger.update(reconstruction_error_score(x, x_hat), DT_S)
    rip = classifier.update(tracker.step(pipeline.convert(pair).l))
    engine.update(drowning, rip)

    return relayed, (relay_times[0] - t0) if relay_times else None


def _forbidden_socket(*args, **kwargs):
    raise AssertionError("frame-to-relay alert path attempted to open a network socket")


class TestAlertPathDoesNotTouchTheNetwork(unittest.TestCase):
    def _run_offline(self):
        with mock.patch.object(socket, "socket", side_effect=_forbidden_socket), \
                mock.patch.object(socket, "create_connection", side_effect=_forbidden_socket):
            return _run_frame_to_relay()

    def test_alert_fires_with_networking_disabled(self):
        relayed, _ = self._run_offline()
        self.assertEqual([a.kind for a in relayed], ["drowning"])

    def test_frame_ingress_to_relay_latency_under_budget(self):
        _, latency_ns = self._run_offline()
        self.assertIsNotNone(latency_ns, "relay was never invoked")
        latency_ms = latency_ns / 1e6
        print(
            f"\n[ocean-vision-v1-1ou] frame ingress -> relay: {latency_ms:.1f} ms "
            f"(dev machine CPU, {SIDE}x{SIDE}x{WINDOW} frames, untrained AE; not edge-node/1080p evidence)"
        )
        self.assertLess(latency_ms, LATENCY_BUDGET_MS)


if __name__ == "__main__":
    unittest.main()
