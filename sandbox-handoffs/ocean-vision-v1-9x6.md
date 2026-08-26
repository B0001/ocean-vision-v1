# Handoff: ocean-vision-v1-9x6 — Dual-sensor frame ingress + 120-frame ring buffer

**Status:** closed. Acceptance criteria met by code + tests below; nothing in
this container can validate it against real cameras or real ocean footage.

## What I changed

First code in the repo — no `pyproject.toml`, `src/`, or `tests/` existed
before this session. Created:

- `pyproject.toml` — `uv init --bare` + `uv add numpy` (`numpy>=2.5.2`, the
  only new dependency). Added `[build-system]`/hatchling config so
  `src/ov1` installs editable and imports as `ov1` from tests.
- `src/ov1/ingress/frame.py` — `Frame`: one sensor's captured frame
  (`sensor_id`, `frame_index`, `timestamp_ns`, `data: np.ndarray`).
  Validates `frame_index >= 0` and `timestamp_ns >= 0`.
- `src/ov1/ingress/source.py` — `FrameSource` (ABC) and
  `SyntheticFrameSource` (in-memory replay, tests/dev only).
- `src/ov1/ingress/sync.py` — `DualSensorSynchronizer`: pairs RGB/NIR frames
  by hardware timestamp within a skew tolerance; `SyncedFramePair`;
  `SensorSyncError`; `SensorStarvedError`.
- `src/ov1/ingress/ring_buffer.py` — `RingBuffer`: fixed-capacity rolling
  window of `SyncedFramePair`s, default capacity 120; `dropped_frame_count`,
  `pushed_frame_count`, `is_full`; `FrameGapError`; `FrameReorderError`.
- `src/ov1/ingress/__init__.py` — public re-exports.
- `tests/test_ingress_source.py`, `test_ingress_sync.py`,
  `test_ingress_ring_buffer.py`, `test_ingress_integration.py`,
  `test_ingress_no_network.py`.

## Test-run output (verbatim)

```
$ uv run python -m unittest discover tests -v
...
----------------------------------------------------------------------
Ran 17 tests in 0.002s

OK
```

All 17 pass. This is the command and output that justifies every claim
below about behavior — I did not run anything against a camera or a GPU.

## How the acceptance criteria are met

> Two synchronized streams decoded; sustained 120-frame window with frame
> drop counter; <1 frame skew between sensors.

- **Two synchronized streams decoded** — `DualSensorSynchronizer.next_pair()`
  pulls one `Frame` from each of an RGB and an NIR `FrameSource` and returns
  a `SyncedFramePair` only when they're within tolerance.
  `test_ingress_sync.py::test_pairs_frames_within_tolerance` and
  `test_ingress_integration.py::test_clean_streams_sustain_a_120_frame_window_with_no_drops`.
- **Sustained 120-frame window with frame drop counter** —
  `RingBuffer(capacity=120)` holds a rolling window (`collections.deque`);
  `dropped_frame_count` accumulates from two independent sources of drop: the
  synchronizer's own counter (a sensor frame discarded to regain alignment)
  and the ring buffer's own frame-index gap detector (defense in depth, in
  case something pushes into the buffer without going through the
  synchronizer). `test_ingress_integration.py::test_window_still_sustains_120_frame_window_after_a_missing_nir_frame`
  drives a stream with one NIR frame physically missing all the way to a full
  120-frame window and asserts `sync.dropped_frame_count == 1`.
- **<1 frame skew between sensors** — tolerance is computed as
  `1 frame period * max_skew_frames` (default `max_skew_frames=1.0`), and the
  comparison is `abs(skew_ns) >= tolerance_ns` to reject, i.e. only skew
  strictly less than one frame period is accepted. Every synced-pair test
  asserts `abs(pair.skew_ns) < sync.skew_tolerance_ns`.

## Fail-loud, never-silent (per this repo's standing rules)

- A skew violation raises `SensorSyncError` immediately — the lagging
  sensor's frame is dropped and counted *before* the exception is raised, and
  the caller must explicitly call `next_pair()` again to retry. Nothing
  substitutes a stale or duplicate frame to keep the stream moving.
- A starved source (one sensor produced no frame while the other did) raises
  `SensorStarvedError` rather than returning a partial or default pair.
- A frame-index gap pushed into the ring buffer raises `FrameGapError` — the
  buffer still appends the new frame (so the window keeps advancing and
  doesn't get stuck), but the caller cannot process it without seeing the
  exception. An out-of-order/duplicate index raises `FrameReorderError`
  *without* appending — history is never silently rewritten.

## No-network guarantee for this leg of the ingress -> alert path

`tests/test_ingress_no_network.py` monkeypatches `socket.socket` and
`socket.create_connection` to raise `AssertionError` if called at all, then
runs the full synchronizer + ring-buffer pipeline to a full 120-frame window
through that patch. It passes, i.e. this code makes zero socket calls. This
only covers ingress; inference and alerting are separate beads with their own
version of this test still to write.

## Constants introduced

| Name | Value | Unit | Source |
| --- | --- | --- | --- |
| `SPEC_TEMPORAL_WINDOW_FRAMES` (ring buffer default capacity) | 120 | frames | Spec section 2, "Temporal Window" row, and this bead's acceptance criteria verbatim. Architectural constant, not a measurement. |
| `nominal_frame_rate_hz` | *(none — required constructor arg)* | Hz | Not hardcoded. Caller must supply the installation's actual capture rate. Spec 4.1 targets 60 FPS; tests use 60.0 as an example, not a default baked into the library. |
| `max_skew_frames` | 1.0 (default, overridable) | frame periods | My reading of this bead's own acceptance criterion ("<1 frame skew"), not a spec citation — flagging as **my choice**, not a spec value. A site could tighten or loosen it via the constructor argument. |

No `tau_drowning`, homography, shoreline normal, or 850nm gain constants are
touched by this bead — those belong to later stages and are correctly out of
scope here.

## Spec target this bead's code touches, and what does/doesn't measure it

- **Temporal Window ≥ 60 frames, target 120 frames (spec section 2)** — the
  ring buffer's frame-count capacity satisfies the *architectural* shape of
  this requirement (it can hold and sustain 120 frames). Nothing here
  measures whether 120 frames of *real* 30 or 60 FPS footage is sufficient
  "to capture the full frequency of active drowning responses" — that's an
  empirical claim about human drowning dynamics this container cannot touch.
- **Ingress Resolution 4K @ 60 FPS (spec section 2 / 4.1)** — not implemented
  or measured. `Frame.data` is an arbitrary `np.ndarray`; nothing in this
  bead enforces or checks resolution or true capture rate, because there is
  no camera to capture from. A real V4L2/NIR backend would need to assert
  the negotiated mode matches, and that assertion doesn't exist yet.
- **Inference Latency, FNR, FPR, Power Envelope** — not touched by this bead
  at all.

## Spec inconsistency flagged (not resolved)

Spec section 2's "Temporal Window" row says **"120 frames @ 30 FPS"**, but
section 4.1 mandates ingress at **4K/60 FPS**. 120 frames is 4.0s at 30 FPS
but only 2.0s at 60 FPS — a 2x difference in how much history the window
actually represents. This bead's acceptance criteria only say "120-frame
window," so I built the ring buffer as **frame-count-based, not
duration-based**, sidestepping the disagreement rather than resolving it.
Whoever owns spec 2 vs 4.1 should reconcile which frame rate is authoritative
before the temporal-window size is treated as a fixed number of *seconds* by
the inference stage (spec 4.2's input tensor is `T=120`, same ambiguity
inherited downstream).

## What I decided not to do, and why

- **No real capture backend (V4L2 / GStreamer / NIR driver).** No camera, no
  GPU, no hardware in this container — building one untestable against real
  hardware would be code nobody can verify, and risks silently encoding wrong
  assumptions about a driver I've never run. `FrameSource` is designed as the
  extension point; a follow-up bead with real hardware access should
  implement a concrete subclass and hardware-loopback-test the timestamp
  source specifically (kernel buffer timestamp vs PTP vs software).
- **No DoLP / Stokes-parameter computation, no Lab-space normalization.**
  Explicitly out of scope for this bead (they're `ocean-vision-v1-bze` and
  `ocean-vision-v1-3qj`, both currently blocked on this bead and now
  unblocked). `Frame.data` is left as a raw per-sensor tensor; those beads
  define what "processed" data looks like downstream of the ring buffer.
- **No frame-rate default baked into `DualSensorSynchronizer`.** Per this
  repo's standing rule against hardcoding installation properties, the
  constructor requires `nominal_frame_rate_hz` explicitly rather than
  defaulting to the spec's 60 FPS target — a default here would silently
  become wrong the day a site runs at a different confirmed rate.
- **No bounded retry limit inside `next_pair()`.** On a `SensorSyncError` I
  resync by dropping exactly one lagging frame and letting the caller retry.
  I didn't add an internal "give up after N attempts and raise fatal" policy
  because that's an alerting/degradation-mode decision (spec 5's "degrade
  visibly" clause) that belongs to whatever owns the ingest loop, not to this
  synchronizer — baking a retry cap in here would be a policy choice
  disguised as a mechanism.

## What I could not verify

- Real sensor timestamp behavior — whether V4L2 buffer timestamps or a PTP
  clock actually deliver sub-frame-period precision on the target hardware.
  `timestamp_ns` is treated as trustworthy input; if the real driver's clock
  is noisier than assumed, the skew tolerance may need per-site tuning (the
  constructor already exposes `max_skew_frames` for that).
- Whether a 120-frame / 2-4s window is actually sufficient signal for
  detecting drowning dynamics — that's the spec-inconsistency question above,
  and it's a biomechanics/data question, not a code question.
- Sustained throughput at 4K/60 on real hardware (memory bandwidth, copy
  overhead of `np.ndarray` frames in the ring buffer) — nothing in this
  container has anything resembling the target Jetson AGX Orin, and I did
  not report any timing numbers because they would mean nothing here.

## Git status — tree left ready to commit, not committed

Per this repo's conservative git policy, nothing was committed or pushed.

```
$ git status --short
 M .beads/issues.jsonl
 M .gitignore              (pre-existing, not from this bead)
 M AGENTS.md                (pre-existing, not from this bead)
D  iterate-beads.sh         (pre-existing, not from this bead)
?? .claude/settings.local.json   (pre-existing, not from this bead)
?? pyproject.toml
?? sandbox-prompt.md        (pre-existing, not from this bead)
?? src/
?? tests/
?? uv.lock
```

Suggested commands for a human to run:

```
git add pyproject.toml uv.lock src/ tests/ .beads/issues.jsonl
git commit -m "ocean-vision-v1-9x6: dual-sensor ingress + 120-frame ring buffer"
```

The `.gitignore`/`AGENTS.md`/`iterate-beads.sh`/`.claude/settings.local.json`/
`sandbox-prompt.md` changes predate this session and are not mine to stage or
explain.

## Bead status

Closed `ocean-vision-v1-9x6` — acceptance criteria are met by the code and
tests above. This unblocks `ocean-vision-v1-bze` (DoLP specular glare mask)
and `ocean-vision-v1-3qj` (Lab-space color normalization), both of which can
now consume `Frame`/`SyncedFramePair`/`RingBuffer` as their upstream source.
