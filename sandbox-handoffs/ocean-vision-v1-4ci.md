# ocean-vision-v1-4ci — Baseline-only training dataset and DataLoader

## Status: closed (evidence below)

## What I found

This bead was left `in_progress` by a prior session (claimed 2026-08-11
11:09:45). Its deliverables — `src/ov1/data/manifest.py` (195 lines) and
`src/ov1/data/dataset.py` (249 lines) — were already fully implemented and
had already been committed, but bundled into an unrelated commit
(`62c7328`, titled for a different bead, `ocean-vision-v1-bag`). On top of
that committed base, the working tree had further uncommitted changes
(`src/ov1/data/__init__.py` exporting `WindowBatch`/`collate_window_samples`)
and two untracked test files (`tests/test_data_dataset.py`,
`tests/test_data_manifest.py`, 43 tests total) — this is the actual 4ci work,
just never closed out.

Two other beads' close notes (`ocean-vision-v1-dxj`, `ocean-vision-v1-xe3`)
independently flagged "1 pre-existing unrelated failure in
ocean-vision-v1-4ci's uncommitted work" without fixing it. I found and fixed
that failure (see below) — it was a test bug, not an implementation bug.

## What I changed

One line, in `tests/test_data_dataset.py`:
`test_non_overlapping_windows_by_default` called `ds._index_start(i)`, a
method that was never implemented anywhere in `BaselineWindowDataset`. The
dataset's real internal state is `self._index: list[tuple[int, int]]` —
`(clip_index, start_frame)` pairs, built and consumed correctly by
`__getitem__` and every other passing test. I fixed the test to read that
existing attribute directly instead of inventing a method on the production
class just to satisfy one whitebox test:

```python
starts = sorted(ds._index_start(i) for i in range(len(ds)))   # before: AttributeError
starts = sorted(start for _clip_index, start in ds._index)    # after
```

No production code (`dataset.py`, `manifest.py`, `data/__init__.py`) needed
any change — reading both files end-to-end, they already satisfy the bead's
acceptance criteria as written.

## Acceptance criteria vs. evidence

- **"Dataset yields correctly shaped windows"** — `BaselineWindowDataset`
  (`src/ov1/data/dataset.py:203`) assembles windows through the same
  `InferenceTensorAssembler` the real-time inference path uses
  (ocean-vision-v1-3fy), so training and inference windows share one code
  path. Verified by
  `tests/test_data_dataset.py::TestBaselineWindowDatasetShapeAndIndexing`
  (shape, default spec-4.2 shape, non-overlapping/overlapping stride
  indexing, short-clip skip-not-raise) and `TestDataLoaderIntegration`
  (DataLoader + `collate_window_samples` batching, since plain
  `default_collate` can't batch the `WindowSample` dataclass — proven by
  `test_default_collate_rejects_window_sample`).
- **"documented provenance/exclusion criteria per clip"** —
  `ClipProvenance` (`src/ov1/data/manifest.py:53`) requires `clip_id`,
  `path`, `label`, `source`, `site_id`, `capture_date`, `reviewed_by`,
  `split`, with optional `exclusion_reason`/`notes`; `load_manifest` raises
  `ManifestError` (fail-loud, not skip/default) on any missing/malformed
  field. `gate_baseline_clips` is an allow-list: only `label == "baseline"`
  AND no `exclusion_reason` is included; everything else is excluded with a
  concrete, auditable reason (`GatedManifest.excluded` — never a silent
  drop). Verified by `tests/test_data_manifest.py::TestLoadManifest` and
  `::TestGateBaselineClips` (18 tests, including case-sensitivity of the
  label and mixed-corpus partitioning).
- **"held-out baseline split reserved for threshold calibration"** —
  `GatedManifest.included_train` / `.included_held_out` partition strictly
  by `split`; `BaselineWindowDataset` is split-agnostic (it builds from
  whatever list it's handed), so isolation is enforced at the call site by
  which property the caller passes. Verified by
  `tests/test_data_manifest.py::TestGatedManifestSplits` and
  `tests/test_data_dataset.py::TestHeldOutSplitIsolation` (end-to-end: gate
  a mixed train/held_out corpus, build a dataset from
  `gated.included_train`, assert the held-out clip_id never appears).

All three criteria are met by code already in the tree; I did not need to
add functionality, only fix a test that was checking a nonexistent API.

## Test run (verbatim)

Full suite:
```
$ UV_PROJECT_ENVIRONMENT=/tmp/venv uv run python -m unittest discover tests
...................................................................................................................................................................................
----------------------------------------------------------------------
Ran 179 tests in 3.915s

OK
```

Dataset/manifest-specific (this bead's own tests):
```
$ UV_PROJECT_ENVIRONMENT=/tmp/venv uv run python -m unittest tests.test_data_dataset tests.test_data_manifest -v
...
----------------------------------------------------------------------
Ran 43 tests in 0.201s

OK
```

## Constants introduced

None by me. Pre-existing in the module (both categorical, not spec numeric
targets):
- `REQUIRED_BASELINE_LABEL = "baseline"` (`manifest.py:33`) — the allow-list
  label string. A naming convention, not a calibration number.
- `VALID_SPLITS = ("train", "held_out")` (`manifest.py:40`) — the two split
  names this bead's own acceptance criteria specify.
- Window/tensor shape defaults (`SPEC_HEIGHT`, `SPEC_WIDTH`,
  `SPEC_TEMPORAL_WINDOW_FRAMES`, `SPEC_CHANNELS`) come from
  `ov1.tensor.assemble`, established under ocean-vision-v1-3fy, not
  reintroduced here.

## Spec performance targets touched

None. FNR < 0.01%, FPR < 2.0%, and the 250ms inference latency target (spec
2) are untouched by this bead — it's training-data plumbing, not inference
or the alert path. Nothing here measures any of them, and nothing in this
container could.

## Network / offline

`dataset.py` and `manifest.py` do not import `socket`, `urllib`,
`requests`, or any HTTP/hub client — `grep` confirms zero matches. This is
offline training-time tooling (reads local `.npz` files and a local JSONL
manifest), not on the ingress → inference → alert path spec 5.2 requires to
stay network-free, so I did not add a dedicated no-network test for it —
that requirement applies to the runtime alert path, and this code has no
network surface to test against in the first place.

## What I decided not to do, and why

- **Did not add a video-decode path from raw footage to `.npz`.** The
  module docstring (`dataset.py:12`) is explicit that this is out of scope
  — `NpzClipFrameSource` starts from already-preprocessed `FrameChannels`
  planes; producing those from real footage is offline tooling this bead
  doesn't build. I did not expand scope to build it.
- **Did not curate an actual corpus of baseline clips.** There is no ocean
  footage of any kind in this container (per the standing session
  instructions) — "curate a corpus" can't be executed here. What's built is
  the mechanism (manifest schema, gate, dataset, DataLoader) a real corpus
  would be loaded through; the corpus itself is a data-collection task, not
  a coding one.
- **Did not squash the pre-existing commit's bundling of this bead's code
  with an unrelated bead's title.** Rewriting `62c7328`'s history was out
  of scope for this session (git policy: no commits at all this session)
  and would be a separate, riskier act of history surgery a human should
  decide on, not silently do.
- **Did not touch `src/ov1/model/rip_current.py` or the `model/__init__.py`
  diff sitting in the working tree.** Those are ocean-vision-v1-xe3's
  already-closed deliverable, left uncommitted per that bead's own
  handoff — unrelated to this bead's scope, so I left them exactly as
  found.

## What I could not verify

- Real-footage behavior of any kind — no camera, no ocean clips, no GPU in
  this container. Everything above is validated against small synthetic
  tensors (6-8 px frames, a handful of windows), per the standing
  instructions: this measures the code's correctness on synthetic data, not
  the ocean.
- Whether a real curated corpus, once it exists, will actually be
  contamination-free — the gate enforces the *label* is "baseline" and
  unexcluded; it cannot itself detect mislabeled footage. That's a human
  review process question, not something code can close.

## Commands for a human to run

```bash
git add src/ov1/data/__init__.py src/ov1/data/dataset.py tests/test_data_dataset.py tests/test_data_manifest.py
git commit -m "Baseline-only training dataset and DataLoader (ocean-vision-v1-4ci)"
# Note: src/ov1/data/manifest.py is already committed as-is (in 62c7328,
# under an unrelated commit message) -- nothing to stage there.
# src/ov1/data/dataset.py DOES have an uncommitted diff on top of that base
# (WindowBatch, collate_window_samples, and the output_shape property) --
# `git diff src/ov1/data/dataset.py` before committing to see exactly what's
# new versus what 62c7328 already has.
```
