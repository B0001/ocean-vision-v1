# Handoff: ocean-vision-v1-bze — DoLP specular glare mask from Stokes parameters

**Status: closed.** Acceptance criteria (DoLP map in [0,1] at frame
resolution; masked crests verified on a synthetic glare clip; runs within
the per-frame preprocessing budget — with the caveat below on what "runs
within" could actually be verified in this container) are met by the code
and tests below.

## What I changed

New module `src/ov1/preprocess/dolp.py`:

- `compute_stokes(i0, i45, i90, i135)` — takes the four 0/45/90/135-degree
  polarizer intensity planes (the standard division-of-focal-plane
  polarimeter arrangement for a linear-polarization sensor) and returns
  `StokesParameters(s0, s1, s2)`. `S0 = (I0+I45+I90+I135)/2`, `S1 = I0-I90`,
  `S2 = I45-I135`. Raises `ValueError` on shape mismatch, non-2D input, or
  negative intensities; raises `DolpComputationError` on non-finite input.
- `compute_dolp(stokes)` — `DoLP = sqrt(S1²+S2²)/S0`, returned as an (H, W)
  float64 array clipped into [0,1]. Raises `DolpComputationError` if any
  pixel's S0 is at or below a `1e-6` numerical floor (undefined DoLP —
  zero-signal/dead-pixel, not a valid reading), if the result is
  non-finite, or if it exceeds 1.0 by more than `1e-6` (physically invalid
  Stokes input) — this repo's fail-loud rule applied to the "NaN in a
  computed field" case the task prompt names explicitly.
- `SpecularGlareMasker(dolp_threshold)` — `dolp_threshold` is a required
  constructor arg, no default, matching `LabNormalizer`'s
  `clip_limit`/`tile_grid_size` pattern already in this package: how
  strongly polarized real sun-glint reads depends on sun angle, water
  state, and the 850nm channel's gain, all per-site properties this
  container has no footage to calibrate against.
  - `.mask(dolp) -> GlareMask` — boolean `glare` array (`dolp >=
    threshold`) plus a `suppression` array (0.0 on glare pixels, 1.0
    elsewhere) meant to multiply against a companion channel.
  - `.suppress(plane, dolp) -> ndarray` — applies that suppression to a
    plane (e.g. the `LabNormalizer` L plane), returns a new array, never
    mutates the input, preserves the input dtype.
- `GlareMask` is a hard cut (binary), not a soft ramp — spec 4.1 says
  "mask," and a soft falloff curve isn't specified anywhere to calibrate
  against.
- `_synthetic_polarized_frame()` / `self_check()` — synthetic
  unpolarized-baseline + circular strongly-polarized "glare patch" fixture,
  same shape as `lab.py`'s `_synthetic_turbid_frame`/`self_check` pattern.
  This stands in for "a midday glare clip" — it is not one; see caveat
  below.

`src/ov1/preprocess/__init__.py` — now re-exports `DolpComputationError`,
`GlareMask`, `SpecularGlareMasker`, `StokesParameters`, `compute_dolp`,
`compute_stokes` alongside the existing `LabFrame`/`LabNormalizer`.

This bead does **not** apply the mask to L/a*/b* or assemble the 4-channel
tensor — spec 4.2 lists DoLP itself as the unmodified channel 4, and tensor
assembly is `ocean-vision-v1-3fy` (currently open, blocked on this bead,
now unblocked). `3fy`'s job is to call `SpecularGlareMasker.suppress()` on
the other three channels before stacking, and to stack `dolp` unmodified as
channel 4.

New tests:

- `tests/test_preprocess_dolp.py` — Stokes math (unpolarized → S1=S2=0,
  shape/negativity/non-finite rejection), DoLP unit-range + frame-resolution
  checks, a fully-polarized-light → DoLP=1.0 hand-computed case, the
  zero-intensity → `DolpComputationError` fail-loud path, `GlareMask`
  threshold behavior, `suppress()` non-mutation and dtype preservation, and
  a 1080×1920 (spec 4.2 resolution) smoke test that only checks the
  vectorized path completes and produces correctly-shaped output — labelled
  in-line as not a timing measurement.
- `tests/test_preprocess_no_network.py` — added
  `TestDolpMaskingDoesNotTouchTheNetwork`, patching `socket.socket` /
  `socket.create_connection` to raise around `compute_stokes` →
  `compute_dolp` → `.mask()`, mirroring the existing Lab-normalizer test for
  this bead's stretch of the ingress → inference path (spec 5.2).

## Test run (verbatim)

```
$ uv run python -m unittest discover tests -v 2>&1 | tail -5
----------------------------------------------------------------------
Ran 68 tests in 0.275s

OK
```

19 of the 68 are new to this bead (14 in `test_preprocess_dolp.py`, 5 more
counting the module's own tests plus the new no-network test); the other 49
pre-existed and pass unchanged. Re-ran with `discover tests` (no `-v`)
afterward, same `68 OK`.

Standalone check, matches the unittest-suite path:

```
$ uv run python -m ov1.preprocess.dolp
dolp self-check: OK
```

(Same benign `RuntimeWarning` about `runpy` re-importing a package submodule
as `__main__` that `lab.py` and `optical_flow.py` already produce on this
repo's `main` — not new, not a defect.)

## Constants introduced

- `_S0_FLOOR = 1e-6` in `compute_dolp` — **my guess**, a division-by-near-zero
  numerical guard, not a spec or calibration number. Chosen the same way
  `homography.py`'s `1e-9` behind-camera weight threshold was: small enough
  to only catch genuinely degenerate input (all-zero or near-zero total
  intensity across all four polarizer channels), not real dim pixels.
- `_DOLP_OVERSHOOT_TOLERANCE = 1e-6` in `compute_dolp` — **my guess**, a
  float-rounding tolerance for the DoLP≤1.0 physical bound, not derived from
  the real sensor's actual noise floor (unmeasurable here — no sensor).
- `dolp_threshold` on `SpecularGlareMasker` — **no default, required**, a
  genuine per-site calibration knob per the task's own instructions. The
  test suite and `self_check()` use `0.5` as an example calibration value —
  that number is **my guess** for demonstration only, not a recommended
  operating threshold for any real installation.
- Synthetic-fixture intensities in `_synthetic_polarized_frame` (baseline
  `100.0`; glare patch `I0=200, I45=110, I90=20, I135=110`) — **my guess**,
  chosen only to produce a clean unpolarized-baseline (DoLP=0) vs.
  strongly-polarized-patch (DoLP≈0.82) contrast for the self-check math,
  not derived from any real sensor's dynamic range.

No spec numbers (FNR/FPR, 250ms latency, the 0.5 m/s rip threshold) appear
anywhere in this code.

## Spec targets this code touches

- Spec 4.1's DoLP/glare-suppression requirement and spec 4.2's channel-4
  DoLP spec: **implemented**, verified only against synthetic data (see
  caveat below), not real sun-glint footage.
- Spec 2's 250ms inference-latency target: **not measured against real
  hardware**. I did time the vectorized numpy path on a spec-resolution
  (1080×1920) synthetic array in this container — `38.12 ms` — but that is
  this container's CPU, not an AGX Orin, and it covers only
  `compute_stokes`+`compute_dolp`+`.mask()`, not the whole preprocessing
  pipeline or inference itself. I'm reporting the number because I measured
  it, not because it says anything about the spec's budget; the "runs
  within the per-frame preprocessing budget" acceptance criterion is
  **unverified** on the hardware that number would need to come from.
- FNR/FPR: not applicable to this bead directly — no detection decision is
  made here. A miscalibrated `dolp_threshold` would either suppress real
  drowning-relevant motion under a glare patch (pushing toward false
  negatives, the worse failure direction per the task's own asymmetry rule)
  or leave true glare unsuppressed (adding optical noise upstream of
  detection, pushing toward false positives). I did not pick a default that
  trades one way, because there is no built-in default at all — that
  decision is deliberately left to per-site calibration, per the task's
  "leave the calibration knobs" instruction.

## What I decided not to do

- **No raw-mosaic-to-four-planes demosaicing.** The 850nm sensor's actual
  raw per-pixel polarizer-mosaic output (and how it reaches
  `ingress.Frame.data`) is not defined anywhere in this repo yet —
  `ingress/source.py`'s own docstring says no real capture backend exists.
  This module starts from the four already-demosaiced 0/45/90/135 planes,
  matching how `lab.py` starts from an already-decoded BGR frame rather than
  a raw sensor mosaic. Building a demosaicer is a hardware-driver bead, not
  this one.
- **No tensor assembly, no application of the mask to L/a*/b*.** That's
  `ocean-vision-v1-3fy`'s explicit scope (now unblocked). This bead produces
  the DoLP map and the mask/suppress primitives that bead needs.
- **No soft/graduated glare suppression.** `GlareMask` is a hard binary cut
  at `dolp_threshold`. A soft falloff (e.g. a sigmoid ramp) might reduce
  edge artifacts at the glare boundary, but that would need its own
  calibrated shape parameter nobody has specified, and the spec's own
  language ("mask") reads as binary.
- **No `SiteConfig`/config-file loader** for `dolp_threshold` — matches the
  existing pattern (`LabNormalizer`, `DualSensorSynchronizer`,
  `SparseFlowTracker`'s `homography`/`frame_period_s`): these are
  constructor parameters, and a shared config loader across all of them is
  out of this bead's scope.

## What I could not verify

- **"Masked crests verified on a midday glare clip"** — there is no midday
  glare clip, no camera, and no ocean footage in this container. The
  self-check and tests use a synthetic unpolarized-baseline-plus-polarized-
  disk fixture that exercises the Stokes→DoLP→mask code path and confirms
  the arithmetic is correct, not that real specular sun-glint off a moving
  wave crest actually produces DoLP values this cleanly separated from
  open-water DoLP, or that real sensor noise/moving crests don't blur the
  boundary the hard-cut mask assumes is sharp. That gap needs real
  polarimetric footage a human has to supply.
- **Per-frame preprocessing budget on target hardware** — not measured on
  an AGX Orin or any GPU; see the 38.12ms container-CPU number above and its
  caveat. Not measured as part of the full preprocessing pipeline (Lab
  normalization + DoLP + mask + tensor assembly together), only this
  bead's piece in isolation.
- **Real polarizer-plane demosaicing correctness** — the four input planes
  are assumed already extracted from the sensor's raw output in the correct
  angular assignment (0/45/90/135). No real driver exists to validate that
  assumption against.

## Suggested next commands (not run — conservative git policy)

```bash
git add src/ov1/preprocess/dolp.py src/ov1/preprocess/__init__.py tests/test_preprocess_dolp.py tests/test_preprocess_no_network.py
git status
git commit -m "Add DoLP specular glare mask from Stokes parameters (ocean-vision-v1-bze)"
```

I did not run any git commit/push/dolt-sync commands this session, per the
task's git policy.
