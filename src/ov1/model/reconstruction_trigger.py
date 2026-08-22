"""Reconstruction-error scoring and sliding-window continuity trigger for
Head A's drowning-detection alarm (spec 4.2, bead ocean-vision-v1-wtg).

Spec 4.2, verbatim: "The Reconstruction Error Score is calculated as
S_r = ||X - X_hat||^2. An alert is triggered if S_r > tau_drowning
continuously over a sliding temporal window Delta t >= 3.0s." X here is one
window's `SpatiotemporalAutoencoder` input (`ov1.model.autoencoder`, bead
ocean-vision-v1-133); X_hat is that model's `forward()` output on the same
window. This module is downstream of both:

- `reconstruction_error_score()` turns one `(X, X_hat)` pair into the
  scalar S_r spec 4.2 defines.
- `DrowningTrigger` tracks continuity of `S_r > tau_drowning` across
  successive inference steps in wall-clock time (one `update()` call per
  window the pipeline scores, in order), firing only once that continuity
  has held for at least `min_continuous_s` -- the same "accumulate while
  qualifying, reset the instant it doesn't" state machine
  `ov1.model.rip_current.RipCurrentClassifier` uses for spec 4.3's
  "sustained" rip-current requirement (ocean-vision-v1-xe3), applied here to
  spec 4.2's continuity requirement instead.

Calibration knobs (this repo's standing rule -- CLAUDE.md): `tau_drowning`
and `min_continuous_s` are both required `DrowningTrigger` constructor
arguments, with no default.

- `tau_drowning` is explicitly a per-site number in the bead's own
  description ("calibrated per site against held-out baseline") --
  `ov1.data.manifest.VALID_SPLITS`'s `"held_out"` split exists for exactly
  this calibration, and this module does not choose a value for the caller.
- `min_continuous_s` is spec 4.2's own number (3.0s), not a per-site
  physical property -- but it is still a named, overridable argument rather
  than a hardcoded constant, on the same footing as
  `RipCurrentClassifier.min_sustained_s`: nothing in this container has
  measured whether 3.0s is the right continuity window for real thrashing
  dynamics, and a future recalibration effort should not need a code change
  to test a different value.

FNR/FPR (spec 2 targets: <0.01% / <2.0%): `self_check()` below measures a
synthetic false-negative/false-positive rate, but only by replaying
hand-built score traces through `DrowningTrigger` -- there is no ocean
footage, autoencoder training run, or held-out baseline corpus in this
container to measure a real FNR/FPR against. `SyntheticContinuityResult`'s
`synthetic_fnr`/`synthetic_fpr` describe how well this module's continuity
logic matches the synthetic traces it was constructed to pass; per this
repo's standing rule, they must never be read, logged, or reported as
evidence that spec 2's FNR/FPR targets are met.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch


class ReconstructionScoreError(ValueError):
    """Raised when `reconstruction_error_score`'s inputs are malformed -- a
    shape mismatch between `X` and `X_hat`, or a non-finite value in
    either. Per this repo's fail-loud rule, a NaN reconstruction (a
    diverged model, a corrupted frame smuggled in from upstream) must stop
    scoring, not silently produce a score that looks like a real number."""


def reconstruction_error_score(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Spec 4.2's `S_r = ||X - X_hat||^2` for one window: the squared L2
    (Frobenius) norm of the reconstruction residual, summed over every
    element of the window -- batch, channel, time, height, and width. This
    is the literal spec formula (a sum of squares), not a per-element mean;
    `tau_drowning` must be calibrated against scores produced by this exact
    reduction, since mixing a differently-normalized score (e.g. mean
    squared error, which shrinks as window size grows) with a tau
    calibrated against this function's output would silently miscalibrate
    the trigger.

    Accumulates in float64 regardless of `x`'s dtype so summing millions of
    elements (spec 4.2's window is (1,4,120,1080,1920) ~= 1e9 elements)
    doesn't lose precision to float32 accumulation error.

    Raises `ReconstructionScoreError` if `x`/`x_hat` shapes disagree, or if
    either contains a non-finite value.
    """
    if x.shape != x_hat.shape:
        raise ReconstructionScoreError(
            f"X and X_hat shapes disagree: {tuple(x.shape)} vs {tuple(x_hat.shape)}"
        )
    if not torch.isfinite(x).all():
        raise ReconstructionScoreError("X contains a non-finite (NaN/inf) value")
    if not torch.isfinite(x_hat).all():
        raise ReconstructionScoreError("X_hat contains a non-finite (NaN/inf) value")

    residual = x.to(torch.float64) - x_hat.to(torch.float64)
    return float(torch.sum(residual * residual).item())


@dataclass(frozen=True)
class DrowningObservation:
    """One `DrowningTrigger.update()` call's result: the raw score fed in,
    whether it was above `tau_drowning`, how long (wall-clock seconds) the
    score has now stayed *continuously* above `tau_drowning`, and whether
    that continuity has reached `min_continuous_s` -- `is_alerting` is the
    actual spec 4.2 alert condition."""

    score: float
    above_threshold: bool
    sustained_s: float
    is_alerting: bool


class DrowningTrigger:
    """Stateful spec 4.2 continuity trigger, driven one reconstruction-
    error score at a time (one call per inference window the pipeline
    scores, in order).

    Usage: construct once per site/session with that site's calibrated
    `tau_drowning`, then call `update(score, dt_s)` once per window's
    `reconstruction_error_score()` result, in order -- `sustained_s` is
    stateful across calls, and a skipped or reordered step will misreport
    how long the score has stayed above threshold.

    `tau_drowning` and `min_continuous_s` are both required, with no
    default (module docstring).

    False-negative/false-positive trade-off (this repo's standing rule -- a
    false negative is a death, a false positive is alarm fatigue): the
    continuity requirement here is *strict* -- any single reading at or
    below `tau_drowning` resets `sustained_s` to zero, rather than
    tolerating a brief dip (e.g. a debounced or leaky-integrator version
    that forgives one low reading inside an otherwise-qualifying window).
    Forgiving dips would trade FNR budget for FPR budget, the wrong
    direction given spec 2's asymmetric cost. This is this module's design
    choice, not a spec-mandated one -- spec 4.2 says only "continuously",
    not how a single sub-threshold reading inside a window should be
    handled; a future calibration effort that finds this too strict (e.g.
    single-frame sensor noise spuriously resetting a real event) should
    change this deliberately, not by this module silently picking leniency.
    """

    def __init__(self, tau_drowning: float, min_continuous_s: float):
        if tau_drowning <= 0:
            raise ValueError(f"tau_drowning must be > 0, got {tau_drowning}")
        if min_continuous_s <= 0:
            raise ValueError(f"min_continuous_s must be > 0, got {min_continuous_s}")
        self._tau_drowning = tau_drowning
        self._min_continuous_s = min_continuous_s
        self._sustained_s = 0.0

    @property
    def tau_drowning(self) -> float:
        return self._tau_drowning

    @property
    def min_continuous_s(self) -> float:
        return self._min_continuous_s

    @property
    def sustained_s(self) -> float:
        return self._sustained_s

    def reset(self) -> None:
        """Clear accumulated continuity, e.g. after a tracking/inference
        gap the caller has independently detected (dropped window,
        pipeline restart)."""
        self._sustained_s = 0.0

    def update(self, score: float, dt_s: float) -> DrowningObservation:
        """Advance the continuity timer by one inference step.

        Raises `ValueError` if `score` is non-finite or `dt_s` is not
        positive -- never substituted with a default, per this repo's
        fail-loud rule for the detection path: a NaN score (an upstream
        `reconstruction_error_score` bug bypassed) or a non-positive/
        desynchronised `dt_s` must stop the pipeline, not be silently
        treated as "no signal."
        """
        if not math.isfinite(score):
            raise ValueError(f"score must be finite, got {score}")
        if dt_s <= 0:
            raise ValueError(f"dt_s must be > 0, got {dt_s}")

        above = score > self._tau_drowning
        if above:
            self._sustained_s += dt_s
        else:
            self._sustained_s = 0.0

        is_alerting = above and self._sustained_s >= self._min_continuous_s
        return DrowningObservation(
            score=score,
            above_threshold=above,
            sustained_s=self._sustained_s,
            is_alerting=is_alerting,
        )


@dataclass(frozen=True)
class SyntheticContinuityResult:
    """Empirical false-negative/false-positive counts from replaying
    hand-built or randomly generated score traces through a fresh
    `DrowningTrigger` per trace.

    This measures `DrowningTrigger`'s continuity logic against the traces
    it was given -- NOT spec 2's FNR/FPR targets, which describe real-world
    drowning detection against real footage. Nothing in this container can
    produce that measurement (module docstring). Report
    `synthetic_fnr`/`synthetic_fpr` as exactly that: synthetic, unvalidated
    against real data.
    """

    num_positive_traces: int
    num_negative_traces: int
    false_negatives: int
    false_positives: int

    @property
    def synthetic_fnr(self) -> float:
        return self.false_negatives / self.num_positive_traces

    @property
    def synthetic_fpr(self) -> float:
        return self.false_positives / self.num_negative_traces


def evaluate_synthetic_traces(
    positive_traces: Sequence[Sequence[float]],
    negative_traces: Sequence[Sequence[float]],
    *,
    tau_drowning: float,
    min_continuous_s: float,
    dt_s: float,
) -> SyntheticContinuityResult:
    """Replay each trace through a fresh `DrowningTrigger` (one trigger per
    trace, so traces don't share continuity state) and count
    false negatives/positives against the caller's *intent* for that
    trace -- a "positive" trace is constructed to represent continuous
    distress-like scores and is expected to alert at some point during
    replay; a "negative" trace is constructed to represent baseline/noise
    scores and is expected never to. Those labels are this function's
    caller's construction, not ground truth from real footage or a trained
    model (module docstring) -- see `self_check()` for how the traces
    themselves are built.

    Raises `ValueError` if either trace list is empty (an FNR/FPR over zero
    traces is undefined, not zero).
    """
    if not positive_traces:
        raise ValueError("positive_traces must not be empty")
    if not negative_traces:
        raise ValueError("negative_traces must not be empty")

    false_negatives = 0
    for trace in positive_traces:
        trigger = DrowningTrigger(tau_drowning=tau_drowning, min_continuous_s=min_continuous_s)
        if not any(trigger.update(score, dt_s).is_alerting for score in trace):
            false_negatives += 1

    false_positives = 0
    for trace in negative_traces:
        trigger = DrowningTrigger(tau_drowning=tau_drowning, min_continuous_s=min_continuous_s)
        if any(trigger.update(score, dt_s).is_alerting for score in trace):
            false_positives += 1

    return SyntheticContinuityResult(
        num_positive_traces=len(positive_traces),
        num_negative_traces=len(negative_traces),
        false_negatives=false_negatives,
        false_positives=false_positives,
    )


def self_check() -> SyntheticContinuityResult:
    """Standalone runnable check (also invoked from the unittest suite).

    1. Hand-built traces confirm the continuity rule's edges exactly: a
       spike solidly above `tau_drowning` for less than `min_continuous_s`
       must not fire; the same spike extended past `min_continuous_s` must;
       and a single sub-threshold reading deep into an otherwise-qualifying
       spike must reset the continuity clock to zero rather than being
       forgiven (this module's strict-continuity design choice, docstring).
    2. A batch of randomly generated synthetic "distress-like" (continuous
       high score) and "baseline-like" (low score with one below-window
       spike) traces are replayed through `evaluate_synthetic_traces()` to
       produce a `synthetic_fnr`/`synthetic_fpr` reading -- reported here
       strictly as measuring this trigger's logic against synthetic data
       (module docstring), NOT as spec 2 compliance evidence.

    Raises `AssertionError` on any failure.
    """
    tau = 1.0
    dt_s = 1.0 / 30.0
    min_continuous_s = 3.0

    # -- Edge 1: a spike shorter than min_continuous_s must not fire.
    trigger = DrowningTrigger(tau_drowning=tau, min_continuous_s=min_continuous_s)
    short_spike_steps = int(2.9 / dt_s)
    for _ in range(short_spike_steps):
        observation = trigger.update(tau + 5.0, dt_s)
        assert not observation.is_alerting, "a spike shorter than min_continuous_s must not fire"

    # -- Edge 2: the same spike, extended past min_continuous_s, must fire.
    fired = False
    for _ in range(int(0.5 / dt_s)):
        observation = trigger.update(tau + 5.0, dt_s)
        if observation.is_alerting:
            fired = True
            break
    assert fired, "a spike sustained past min_continuous_s must fire"

    # -- Edge 3: a single sub-threshold reading resets the clock, even deep
    # into an otherwise-qualifying spike (strict, non-leaky continuity).
    trigger = DrowningTrigger(tau_drowning=tau, min_continuous_s=min_continuous_s)
    for _ in range(int(2.5 / dt_s)):
        trigger.update(tau + 5.0, dt_s)
    observation = trigger.update(tau - 5.0, dt_s)
    assert observation.sustained_s == 0.0, "a sub-threshold reading must reset sustained_s to zero"
    assert not observation.is_alerting
    last_observation = None
    for _ in range(int(2.5 / dt_s)):
        last_observation = trigger.update(tau + 5.0, dt_s)
    assert not last_observation.is_alerting, "the clock must restart from zero after a reset, not resume"

    # -- Synthetic FNR/FPR reading (module docstring: synthetic only).
    rng = torch.Generator().manual_seed(0)
    num_steps = int(5.0 / dt_s)  # 5s traces, comfortably longer than min_continuous_s

    positive_traces = [
        (tau + 3.0 + 0.05 * torch.randn(num_steps, generator=rng)).tolist()
        for _ in range(20)
    ]

    negative_traces = []
    for _ in range(20):
        trace = (0.3 * tau + 0.05 * torch.randn(num_steps, generator=rng)).clamp(min=0.0).tolist()
        # One below-min_continuous_s spike, to exercise the negative
        # class's near-miss edge rather than only quiet baseline noise.
        spike_start = num_steps // 2
        spike_len = int(1.0 / dt_s)
        for i in range(spike_start, spike_start + spike_len):
            trace[i] = tau + 2.0
        negative_traces.append(trace)

    result = evaluate_synthetic_traces(
        positive_traces,
        negative_traces,
        tau_drowning=tau,
        min_continuous_s=min_continuous_s,
        dt_s=dt_s,
    )
    assert result.false_negatives == 0, (
        f"unexpected false negative on an unambiguous synthetic distress trace: "
        f"{result.false_negatives}/{result.num_positive_traces}"
    )
    assert result.false_positives == 0, (
        f"unexpected false positive on a synthetic baseline trace with a below-window spike: "
        f"{result.false_positives}/{result.num_negative_traces}"
    )

    return result


if __name__ == "__main__":
    _result = self_check()
    print(
        "reconstruction_trigger self-check: OK, "
        f"synthetic_fnr={_result.synthetic_fnr:.4f} synthetic_fpr={_result.synthetic_fpr:.4f} "
        "(synthetic data only -- spec 2's <0.01%/<2.0% targets are NOT validated by this number)"
    )
