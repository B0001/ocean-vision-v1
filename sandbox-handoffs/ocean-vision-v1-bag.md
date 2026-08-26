# Handoff: ocean-vision-v1-bag — Sparse optical flow velocity field v(x,y,t)

**Status: closed.** Acceptance criteria (per-window velocity field with
pixel/s -> m/s conversion via site homography; drift-tolerant re-seeding;
sanity check on a clip with known current direction) are met by code + tests
below.

## What I changed

New package `src/ov1/flow/`:

- `homography.py` — `Homography` (frozen dataclass wrapping a (3,3) matrix)
  and `PointBehindCameraError`. `pixel_to_metres()` does the homogeneous
  divide `matrix @ [x,y,1]` -> `(X/w, Y/w)`. Validates shape, finiteness, and
  non-singularity at construction; raises on a homogeneous weight `w < 1e-9`
  rather than returning a blown-up metre position for a point that's fallen
  off the modelled water plane.
- `optical_flow.py` — `SparseFlowTracker`: `seed(l_frame)` runs
  `cv2.goodFeaturesToTrack` on the first L-channel frame;
  `step(l_frame)` runs `cv2.calcOpticalFlowPyrLK` against the previous frame,
  drops points with `status == 0`, maps surviving prev/current pixel
  positions through the `Homography`, and returns metres/second velocity as
  `(curr_m - prev_m) / frame_period_s`. `VelocityField` is the per-step
  output (`positions_px`, `velocity_mps`, `dt_s`).
  - Drift-tolerant re-seeding: when tracked-point count after a step falls
    below `re_seed_fraction * max_corners`, `step()` runs
    `goodFeaturesToTrack` again with a mask excluding a `min_distance` ring
    around every surviving point, and folds the new corners in. Surviving
    points are never reset — only the shortfall is topped up.
  - Fails loud: `NoTrackablePointsError` on a featureless seed frame,
    `FlowFieldError` if every point is lost in a step or the flow output
    contains non-finite pixel positions, plain `ValueError`/`RuntimeError`
    on shape/dtype/call-order misuse. Nothing is silently substituted.
- `__init__.py` — re-exports `Homography`, `PointBehindCameraError`,
  `SparseFlowTracker`, `VelocityField`, `FlowFieldError`,
  `NoTrackablePointsError`.

New tests:

- `tests/test_flow_homography.py` — shape/finiteness/singularity rejection,
  hand-computed affine and perspective mappings, behind-camera rejection.
- `tests/test_flow_optical_flow.py` — self-check, known-drift-direction
  recovery (both signs), re-seeding recovers tracked-point count and
  respects `min_distance` from survivors (whitebox-checked via
  `tracker._prev_points`), and every fail-loud path (blank seed frame,
  `step()` before `seed()`, wrong ndim/dtype, mid-track shape change,
  all-points-lost, non-finite flow output — the last two via
  `unittest.mock.patch` on `cv2.calcOpticalFlowPyrLK` since neither is
  reliably reproducible from real synthetic tracking behavior).
- `tests/test_flow_no_network.py` — patches `socket.socket` /
  `socket.create_connection` to raise, then runs `seed()` + `step()` through
  it, mirroring `test_preprocess_no_network.py`'s pattern for this bead's
  stretch of the ingress -> inference path (spec 5.2).

## Test run (verbatim)

```
$ uv run python -m unittest discover tests -v 2>&1 | tail -5
----------------------------------------------------------------------
Ran 49 tests in 0.123s

OK
```

22 of the 49 are new (this bead); the other 27 pre-existed and still pass
unchanged. Ran the full suite three times in a row to check for flakiness
from the synthetic-clip RNG seed (`np.random.default_rng(42)`, fixed) — same
`49 OK` every time.

Standalone check, matches the unittest-suite path:

```
$ uv run python -m ov1.flow.optical_flow
optical_flow self-check: OK
```

(The `RuntimeWarning` printed alongside that line is a `runpy` quirk from
running a package submodule as `__main__` after its package `__init__` has
already imported it — the same warning `python -m ov1.preprocess.lab`
already produces on this repo's `main`. Not new, not a defect.)

## Constants introduced

- `re_seed_fraction=0.5`, `max_corners=200`, `quality_level=0.01`,
  `min_distance=7.0`, `win_size=(15, 15)`, `max_pyramid_level=2` in
  `SparseFlowTracker.__init__` — **my guess**, conventional OpenCV
  Lucas-Kanade tuning defaults (these are the values in most
  `calcOpticalFlowPyrLK` tutorials/examples), not derived from any spec
  number or calibration. They're algorithm hyperparameters, not per-site
  physical properties, so I judged them safe to default while remaining
  overridable — unlike `homography` and `frame_period_s`, which are
  **required, no default**, because they're literally the site's
  camera-mounting calibration and hardware frame rate (the prompt's own
  example of what must never be baked in).
- `metres_per_pixel = 0.05` and `dx_px_per_frame = 5.0` in `self_check()` /
  the `test_known_drift_recovered_in_metres` test — **my guess**, an
  arbitrary synthetic scale chosen only to make the sanity-check math
  readable by hand. Not a site calibration value; never used outside tests
  and the self-check.
- `1e-9` behind-camera weight threshold in `homography.py` — **my guess**, a
  numerical-stability epsilon, not a spec or calibration number.

No spec numbers (0.5 m/s rip threshold, 15 m channel width, FNR/FPR,
250 ms latency) appear anywhere in this code — they belong to the
downstream classifier (`ocean-vision-v1-xe3`), not this tracker.

## Spec targets this code touches

- Spec 4.3's rip-current velocity threshold (`|v| > 0.5 m/s`) and channel
  width (`< 15 m`) are **not implemented or measured here** — this bead
  produces the `v(x,y,t)` field those thresholds get applied to
  downstream, in `ocean-vision-v1-xe3`. Nothing in this bead's code
  evaluates that threshold.
- Spec 2's 250 ms inference latency: **not measured**. I did not time
  `calcOpticalFlowPyrLK` on anything resembling 4K/60 real footage, and this
  container has no GPU/Jetson to time it on meaningfully anyway. The
  synthetic clips here are 120x160 to 64x64 px, nowhere near 1080p/4K.
- FNR/FPR: not applicable to this bead directly (no detection decision is
  made here), but a wrong or unset homography would silently corrupt every
  downstream m/s value feeding both heads' thresholds — that's why
  `Homography` has no default and validates hard at construction.

## What I decided not to do

- **No point-ID tracking across steps.** The acceptance criteria ask for a
  velocity field and drift-tolerant re-seeding, not persistent per-particle
  identity across the whole window. `VelocityField.positions_px` /
  `velocity_mps` are row-aligned per step, which is enough for both
  downstream heads (autoencoder input tensor assembly, rip classifier
  spatial/magnitude gating) as spec'd. If a later bead needs to track a
  specific foam patch's history across many steps, that's separate scope.
- **No `SiteConfig`/config-file loader.** `Homography` and `frame_period_s`
  are constructor parameters, matching the existing pattern in
  `LabNormalizer` (`clip_limit`, `tile_grid_size`) and
  `DualSensorSynchronizer` (`nominal_frame_rate_hz`) — neither of those
  modules has a config loader either, and inventing one here would be scope
  creep. Whoever wires an actual site config file should build one loader
  used by all of these, not three different ad hoc ones.
- **Did not implement `PolarizationGlareMask` (spec 4.1 DoLP) or the DoLP
  input channel** — out of scope (`ocean-vision-v1-bze`), and this tracker
  only reads the L-plane per its own bead description, so DoLP isn't a
  dependency for it.

## What I could not verify

- Real tracking quality on actual foam/wave footage — no ocean footage in
  this container. The synthetic clips are a rigid-translation textured
  patch (random noise wrapped via `cv2.BORDER_WRAP`), which exercises LK
  tracking, re-seeding, and homography plumbing correctly, but says
  nothing about how well Shi-Tomasi corners land on real breaking-wave foam,
  which is non-rigid, appears/disappears, and isn't uniformly textured.
  That gap is inherent to this container, not something a different test
  design would close.
- Real-world homography calibration procedure/accuracy — no camera, no
  reference markers, no install site. The math is standard and unit-tested
  against hand-computed points, but "does this recover metres correctly for
  an actual mounted camera" needs a real calibration target.
- Timing on Jetson-class hardware, per the 250 ms latency target — not
  measured, not attempted, per this session's standing instructions.
- Behavior on genuinely non-rigid/high-turbidity real flow fields (multiple
  independent foam patches moving differently, occlusion by wave crests) —
  the re-seeding logic is unit-tested for "count drops, re-seed tops it back
  up," not for tracking accuracy under those real-world conditions.

## Suggested next commands (not run — conservative git policy)

```bash
git add src/ov1/flow tests/test_flow_homography.py tests/test_flow_optical_flow.py tests/test_flow_no_network.py
git status
git commit -m "Add sparse Lucas-Kanade optical flow velocity field (ocean-vision-v1-bag)"
```

I did not run any git commit/push/dolt-sync commands this session, per the
task's git policy.
