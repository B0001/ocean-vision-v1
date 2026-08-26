# Handoff: ocean-vision-v1-0h8 — Wire ingress ring-buffer window through Lab/DoLP into inference tensor assembly

Status: **closed**, evidence below. This is a verification handoff, not a
build-from-scratch one: the implementation and its tests already existed in
the working tree and were already committed (in commit `62c7328`, "Add
sparse Lucas-Kanade optical flow velocity field (ocean-vision-v1-bag)" — that
commit bundles several beads' work together, including this one's). The bead
itself was left `in_progress` and never closed. I read the code, checked it
against the bead's description line by line, ran the suite, and am closing it
on that evidence rather than re-implementing anything.

## What exists (verified, not authored this session)

- `src/ov1/pipeline/window_assembly.py` — the glue:
  - `SiteCalibrationConfig` — loads `clip_limit`, `tile_grid_size`,
    `dolp_threshold` from a flat JSON file via `from_json_file()`
    (`SiteConfigError` on missing file, invalid JSON, or a missing required
    field — no fallback default for any of the three, per the "leave the
    calibration knobs" rule). Builds a `LabNormalizer` and a
    `SpecularGlareMasker` from those values.
  - `split_nir_polarization_planes()` — fixes and documents the convention
    that a NIR `Frame.data` is `(H, W, 4)`, planes `[i0, i45, i90, i135]` on
    the last axis. The module docstring is explicit this is a convention
    this bead is choosing (no real 850nm capture driver exists in this repo
    to read the order off of), not a hardware fact.
  - `FrameChannelsPipeline.convert()` — one `SyncedFramePair` →
    `LabNormalizer.normalize(pair.rgb.data)` for L/a*/b*,
    `compute_stokes`/`compute_dolp` on the four split NIR planes for DoLP,
    `FrameChannels.from_lab_and_dolp(lab, dolp, glare_masker=...)` to
    package and apply glare suppression to L. `convert_window()` applies it
    to a sequence, oldest-first.
  - `assemble_tensor_from_window(pairs, pipeline, assembler, pin=...)` — the
    full path: a `RingBuffer.window()` snapshot straight through to
    `InferenceTensorAssembler.assemble()`.
- `tests/test_pipeline_window_assembly.py` (23 tests) — config loading
  (valid/missing-field/missing-file/invalid-JSON), NIR plane splitting
  (correct order, wrong shape/ndim raise), `convert`/`convert_window`
  (matching plane shapes, DoLP in `[0,1]`, glare suppression zeroes L only in
  the glare region, bad NIR shape propagates rather than being absorbed),
  and the full path with a real `RingBuffer` (`buf.window()` →
  `assemble_tensor_from_window()` → spec-shaped tensor; a too-short window
  raises `TensorAssemblyError`).
- `tests/test_pipeline_no_network.py` (1 test) — patches `socket.socket` and
  `socket.create_connection` to raise, drives a full `RingBuffer` push +
  `window()` + `FrameChannelsPipeline` + `InferenceTensorAssembler` pass
  through the patch. This is the network-isolation test spec 5.2 requires
  for anything on the ingress → inference path.

Command whose output backs the "it works" claim:
```
$ uv run python -m unittest discover tests
Ran 110 tests in 0.449s
OK
```
(87 tests existed as of `ocean-vision-v1-3fy`'s handoff; 23 more here bring
it to 110 — matches the file's own test count.)

I also traced every collaborator this module calls against its real
signature to rule out a plausible-but-wrong integration (tests passing on
their own synthetic fixtures doesn't guarantee the wiring to *real*
collaborators is right):
- `RingBuffer.window() -> list[SyncedFramePair]` (`ingress/ring_buffer.py:108`)
- `LabNormalizer.normalize(bgr_frame) -> LabFrame` (`preprocess/lab.py:60`)
- `compute_stokes(i0, i45, i90, i135)`, `compute_dolp(stokes)`
  (`preprocess/dolp.py:56,106`)
- `SpecularGlareMasker.suppress(plane, dolp)` (`preprocess/dolp.py:199`)
- `FrameChannels.from_lab_and_dolp(lab, dolp, glare_masker=None)`
  (`tensor/assemble.py:94`) — matches the optional-masker, no-invented-
  threshold contract `ocean-vision-v1-3fy`'s handoff describes.
All match what `window_assembly.py` calls them with.

## Constants introduced

None by this bead — `window_assembly.py` introduces no new numeric
constants. `NIR_PLANE_ORDER = ("i0", "i45", "i90", "i135")` is a naming/order
**convention**, not a tuned value; the module docstring says so explicitly
and gives the reason (no real hardware driver in this repo to read the order
off of — a documented guess about how the eventual 850nm demosaic driver
will lay out its output). The three real calibration numbers
(`clip_limit`, `tile_grid_size`, `dolp_threshold`) are deliberately **not**
constants in this module — they're per-site JSON config, loaded at runtime,
with no built-in default and a hard failure (`SiteConfigError`) if absent.

## Spec performance targets touched — and what was/wasn't measured

- **FNR < 0.01% / FPR < 2.0% (spec 2):** not touched. This bead does no
  detection; it assembles the tensor detection will later run on.
- **Inference < 250ms (spec 4.2):** not touched. This is pre-inference glue;
  no timing was measured or asserted, on this container or otherwise.
- Nothing in this bead measures any spec accuracy/latency target. All 24
  tests are code-path correctness (shape, dtype, value range, order,
  fail-loud error handling) against small synthetic arrays.

## What I decided not to do, and why

- **Did not re-implement or restructure `window_assembly.py`.** It already
  matched the bead's description exactly (RingBuffer window → Lab/DoLP →
  FrameChannels → assembler, plus a site-config answer) and its own test
  suite plus the collaborator-signature check above gave no reason to doubt
  it. Rewriting working, tested code to "make it mine" would be pure churn.
- **Did not touch the NIR-plane-order convention or try to firm it up
  against a real driver.** No real 850nm capture backend exists in this repo
  (confirmed: `ingress/source.py` has no such driver) — there is nothing to
  verify the convention against yet. It's honestly documented as a guess in
  the module docstring, which is the right amount of certainty for a
  decision with nothing to check it against.
- **Did not add a bundled default site-config file.** Per the "leave the
  calibration knobs" rule, shipping even an "example" `site.json` risks
  someone deploying it unedited on a real beach. `from_json_file` requiring
  an explicit path with no fallback is the safer default.

## What I could not verify

- **Real NIR sensor plane order.** `split_nir_polarization_planes`'s
  `(H, W, 4)` / `[i0, i45, i90, i135]` convention has no real 850nm capture
  driver in this repo to check it against — flagged as a guess in the module
  docstring, and I'm repeating that flag here rather than letting it read as
  settled.
- **Real site calibration values.** `SiteCalibrationConfig` is exercised only
  against synthetic JSON in tests (`clip_limit=2.0`, `tile_grid_size=(8,8)`,
  `dolp_threshold=0.5`) — these are test fixture values, not a recommendation
  for any real site, and nothing here calibrates them against real baseline
  footage.
- **Any latency or accuracy number** — this container has no GPU, no Jetson,
  no ocean footage; nothing above is a measurement of spec 2/4.2 targets.

## Handoff / next commands

The implementation, tests, and bead-relevant tracker updates
(`.beads/issues.jsonl`) are already part of commit `62c7328` on `main`. This
session made no code changes — `git status` is clean apart from an unrelated
untracked `.claude/settings.local.json`. Nothing to commit for this bead.

Full suite (110 tests, 0.449s):
```
$ uv run python -m unittest discover tests
Ran 110 tests in 0.449s
OK
```

No follow-up beads filed — I found no scope gap in this bead's own
description that isn't already covered by an existing bead
(`ocean-vision-v1-3hw` for channel-scale normalization was already filed by
`ocean-vision-v1-3fy`).
