"""Deterministic Alerting Logic Engine (spec 3's pipeline diagram node H,
spec 1 objective 3, spec 5.2; bead ocean-vision-v1-1ou).

Fuses Head A (`ov1.model.reconstruction_trigger.DrowningTrigger`) and Head B
(`ov1.model.rip_current.RipCurrentClassifier`) observations into alerts,
and hands each alert to two injected local sinks: the alarm relay, then the
GPS broadcast.

Fusion rule: logical OR, per head. Spec 3 draws both heads into H as
independent inputs ("Anomaly Score > Threshold", "Rip Current / Threat
Detected") and names no cross-head rule, so either head alerting is enough,
and each head produces its own `Alert`. When both heads reach onset on the
same step, the drowning alert is dispatched first (a person in the water
before a hazard to people who may enter it) -- this module's ordering
choice, not a spec one.

Dedup / re-arm: one alert per onset, not one per step. A head's alert fires
on the step its observation goes from not-alerting to alerting; while it
stays alerting nothing more fires; the first non-alerting observation from
that head re-arms it. Both upstream triggers already reset their continuity
clock on any single sub-threshold reading, so "re-arm" here tracks exactly
their own notion of an event ending. A head that has no new observation on
a step (`None` -- Head A scores once per window, Head B once per flow step,
so they need not tick together) keeps its state unchanged.

A head is marked as latched only *after* both sinks returned. If a sink
raises, the exception propagates and the same onset is re-dispatched on
the next alerting step for that head: a retried alarm is alarm fatigue, a
swallowed one is a false negative (this repo's standing FN-over-FP rule).

Determinism: the engine holds no clock, no randomness, and no I/O of its
own; its only state is a step counter and one latch bit per head. The same
sequence of observations in produces the same sequence of `Alert`s out
(and the same sink calls, in the same order).

Offline (spec 5.2): this module imports only the stdlib and the two head
modules; nothing here opens a socket, and `tests/test_pipeline_alerting_no_network.py`
fails if the frame -> relay path ever does. Telemetry and weight updates
are not on this path and are not handled here.

Hardware gap, stated plainly: this repo runs on a dev machine with no alarm
relay and no GPS/radio hardware. `relay` and `gps_broadcast` are therefore
plain callables the deployment must supply (a GPIO/relay driver, a
VHF-DSC/AIS/LoRa broadcaster, whatever the site uses); no driver for either
exists in this repo, and no stand-in pretends to be one. Nor is the
homography's world plane georeferenced anywhere in this repo, so the
broadcast carries the node's own surveyed position (`node_position`, a
required per-installation input with no default) plus, for rip currents,
channel centroids in the site's local world-plane metres -- not lat/lon of
the hazard itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

from ov1.model.reconstruction_trigger import DrowningObservation
from ov1.model.rip_current import RipCurrentObservation

DROWNING = "drowning"
RIP_CURRENT = "rip_current"


@dataclass(frozen=True)
class NodePosition:
    """The edge node's surveyed WGS84 position, in decimal degrees. A
    per-installation value measured at install time; there is no default."""

    latitude_deg: float
    longitude_deg: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.latitude_deg) and -90.0 <= self.latitude_deg <= 90.0):
            raise ValueError(f"latitude_deg must be in [-90, 90], got {self.latitude_deg}")
        if not (math.isfinite(self.longitude_deg) and -180.0 <= self.longitude_deg <= 180.0):
            raise ValueError(f"longitude_deg must be in [-180, 180], got {self.longitude_deg}")


@dataclass(frozen=True)
class Alert:
    """One alert onset. `step` is the engine's 0-based `update()` count at
    onset. `channel_centroids_m` is the rip channels' world-plane (x, y)
    centroids in metres (empty for a drowning alert: Head A scores a whole
    window, it has no location)."""

    kind: str
    step: int
    node_position: NodePosition
    channel_centroids_m: tuple[tuple[float, float], ...] = ()


AlertSink = Callable[[Alert], None]


class AlertEngine:
    """Stateful but clock-free fusion of Head A/B observations into alert
    onsets (module docstring). Call `update()` once per pipeline step, in
    order."""

    def __init__(self, relay: AlertSink, gps_broadcast: AlertSink, node_position: NodePosition):
        self._relay = relay
        self._gps_broadcast = gps_broadcast
        self._node_position = node_position
        self._step = 0
        self._latched = {DROWNING: False, RIP_CURRENT: False}

    def update(
        self,
        drowning: Optional[DrowningObservation],
        rip: Optional[RipCurrentObservation],
    ) -> list[Alert]:
        """Advance one step. Returns the alerts dispatched this step (empty
        if none). Propagates any exception a sink raises, leaving that head
        un-latched so the next alerting step re-dispatches it."""
        onsets = []
        for kind, alerting, centroids in (
            (DROWNING, None if drowning is None else drowning.is_alerting, ()),
            (
                RIP_CURRENT,
                None if rip is None else rip.is_active,
                () if rip is None else tuple((float(c.centroid_m[0]), float(c.centroid_m[1])) for c in rip.channels),
            ),
        ):
            if alerting is None:
                continue
            if not alerting:
                self._latched[kind] = False
            elif not self._latched[kind]:
                onsets.append(Alert(kind, self._step, self._node_position, centroids))
        self._step += 1

        for alert in onsets:
            self._relay(alert)
            self._gps_broadcast(alert)
            self._latched[alert.kind] = True
        return onsets
