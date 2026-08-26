# Handoff: ocean-vision-v1-3hw — Decide channel-scale normalization for the [L,a*,b*,DoLP] inference tensor

Status: **closed**, evidence below. This bead's implementation was already
present and committed on `main` (commit `62c7328`, "Add sparse Lucas-Kanade
optical flow velocity field") when I claimed the bead this session — a prior
session did the actual work but never wrote a handoff or ran `bd close`. My
job this session was to verify that work is real and correct, not silently
assume it, then close it out.

## What I found (verification, not new code)

`src/ov1/tensor/assemble.py` already makes and documents the decision:

- **Decision: divide L/a\*/b\* (uint8, native `[0, 255]`) by 255.0 to land on
  DoLP's native `[0.0, 1.0]` float range before stacking.** Rejected
  per-channel z-scoring against baseline-footage statistics — the module
  docstring (lines 16–30) gives the reasoning: a `/255` rescale is a fixed
  property of the uint8 encoding (identical at every site), whereas a
  z-score's per-channel mean/std would be a per-site calibration knob this
  container has no baseline footage to fit, which is exactly the kind of
  constant this repo's conventions (`CLAUDE.md`) say must live in a site
  config, not get guessed into general assembly code.
- The scale factor is named (`_UINT8_TO_UNIT_SCALE = 255.0`, line 68) rather
  than a bare literal, with a comment pointing back to this bead.
- Applied in `_assemble_frame` (lines 244–247): `L`/`a`/`b` planes are
  divided by `_UINT8_TO_UNIT_SCALE` after resize; `DoLP` passes through
  unmodified (it's already `[0, 1]`). A comment there notes division commutes
  with `INTER_AREA`/`INTER_LINEAR` resampling (both are weighted averages),
  so scaling after resize vs. before is equivalent up to float rounding.
- `assemble()`'s docstring (lines 179–181) states the postcondition: all four
  channels of the returned tensor are in `[0, 1]`.

I read this end to end for correctness (not just presence of a docstring):
the arithmetic is a plain elementwise divide, DoLP is left untouched, and the
"why /255 not z-score" reasoning holds up against this repo's own
"leave the calibration knobs" rule — a z-score would require a per-site mean
and std this bead has no baseline footage to compute, which is `133`'s
training-loop territory (or a later calibration bead), not this one.

## Evidence the decision is real, not just documented

`tests/test_tensor_assemble.py` (pre-existing, part of the same commit)
exercises the rescale specifically:

- `test_channel_order_matches_source_planes` — asserts `l=10 -> 10/255`,
  `a=20 -> 20/255`, `b=30 -> 30/255`, with a comment naming this bead.
- `test_l_a_b_channels_are_rescaled_onto_dolp_unit_range` — `l=255 -> 1.0`,
  `b=128 -> 128/255`, explicitly checking channels land in DoLP's unit range.
- `test_downscale_matches_direct_cv2_resize` — checks the rescale composes
  correctly with the downscale path (`expected / 255.0`).

Ran the tensor-assembly-specific tests in isolation:
```
$ UV_PROJECT_ENVIRONMENT=/tmp/venv uv run python -m unittest tests.test_tensor_assemble tests.test_tensor_no_network -v
...
Ran 20 tests in 0.251s
OK
```

Ran the full suite:
```
$ UV_PROJECT_ENVIRONMENT=/tmp/venv uv run python -m unittest discover tests
...
Ran 153 tests in 0.587s
FAILED (errors=1)
```

The one failure is **not this bead's**: `test_data_dataset.py` (untracked,
uncommitted in the working tree) calls `ds._index_start(...)`, a method that
does not exist on `BaselineWindowDataset`. That file, and the matching
uncommitted `WindowBatch`/`collate_window_samples`/`output_shape` additions
in `src/ov1/data/dataset.py` and `src/ov1/data/__init__.py`, belong to a
different, separately in-progress bead — `ocean-vision-v1-4ci` ("Baseline-only
training dataset and DataLoader"), whose own module docstring in
`dataset.py` cites `bead ocean-vision-v1-4ci`, not `3hw`. That work was
mid-flight (started, not finished) from a prior session under a different bd
claim (`bd list --status=in_progress` shows `ocean-vision-v1-4ci` still
open, owner `B0001`, assignee `sandbox`). Per this session's scope rule ("do
only this bead" / file, don't fix, out-of-scope work), I left those files
exactly as I found them — I did not edit, stage, or revert them. All tests
that exercise `3hw`'s own code path (`assemble.py`, `test_tensor_assemble.py`,
`test_tensor_no_network.py`) pass.

## Constants introduced

None new. `_UINT8_TO_UNIT_SCALE = 255.0` already existed in the committed
code (source: the uint8 encoding's own range, `2**8 - 1` — not a
measurement, not a per-site calibration number; it's the same constant for
every installation because it describes the pixel format, not the scene).

## Spec performance targets touched

None. This is a preprocessing-scale decision, not a detection or timing
change. Nothing here measures FNR, FPR, or the 250ms inference budget, and
nothing in this bead's diff touches any of those paths.

## What I decided not to do, and why

- **Did not implement z-scoring as an alternative/fallback.** The module
  docstring's reasoning (no baseline footage to fit per-channel stats in
  this container, and per-site stats belong in a site config per this
  repo's conventions) is sound and matches how `LabNormalizer`'s CLAHE
  params and `SpecularGlareMasker`'s `dolp_threshold` are already handled
  elsewhere in this codebase. Re-litigating it would be scope creep on a
  decision that's already made and justified.
- **Did not touch `src/ov1/data/dataset.py`, `src/ov1/data/__init__.py`, or
  the two untracked `tests/test_data_*.py` files.** That's
  `ocean-vision-v1-4ci`'s in-progress work, a separate bead, currently
  claimed and mid-flight. Fixing its `_index_start` gap wasn't this bead's
  job and I was told not to do out-of-scope work in this session.
- **Did not write a new bead for the `4ci` gap.** It isn't undiscovered
  work — `ocean-vision-v1-4ci` already exists, is already open, and already
  covers it; filing a duplicate would just be tracker noise.

## What I could not verify

- **Whether `/255` is actually the right choice for `133`'s training
  dynamics** — that's an empirical question about gradient behavior on real
  Lab+DoLP footage through a Conv3D autoencoder, and this container has no
  GPU, no ocean footage, and no baseline corpus to train on. The module
  docstring already says as much: if `133`'s training shows the model needs
  per-channel standardization instead, that's a transform to add in front of
  the autoencoder, not a reason to revisit this bead.
- **Real Lab/DoLP value distributions** — all evidence here is synthetic
  uint8/float planes (`_synthetic_frame_channels`, the unit tests' fixed
  values), not real footage statistics.

## Handoff / next commands

Working tree is unchanged by this session (verification only, no edits). The
pre-existing uncommitted changes belong to `ocean-vision-v1-4ci`, not this
bead, and are left as found:
```bash
git status
# modified:   .beads/interactions.jsonl, .beads/issues.jsonl (bd state)
# modified:   src/ov1/data/__init__.py, src/ov1/data/dataset.py (ocean-vision-v1-4ci, not this bead)
# untracked:  tests/test_data_dataset.py, tests/test_data_manifest.py (ocean-vision-v1-4ci, not this bead)
# untracked:  .claude/settings.local.json (harness config, not source)
```
Nothing to commit for `3hw` itself — its code (`src/ov1/tensor/assemble.py`,
`tests/test_tensor_assemble.py`) was already committed in `62c7328` before
this session started. This session's only change is the `bd close` below and
this handoff file.

Full suite (153 tests, 1 pre-existing unrelated failure):
```
$ UV_PROJECT_ENVIRONMENT=/tmp/venv uv run python -m unittest discover tests
Ran 153 tests in 0.587s
FAILED (errors=1)
```
Tensor-assembly-scoped suite (this bead's actual evidence, 20 tests):
```
$ UV_PROJECT_ENVIRONMENT=/tmp/venv uv run python -m unittest tests.test_tensor_assemble tests.test_tensor_no_network -v
Ran 20 tests in 0.251s
OK
```
