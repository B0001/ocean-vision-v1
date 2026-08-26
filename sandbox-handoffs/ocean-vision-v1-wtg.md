# ocean-vision-v1-wtg — Reconstruction error scoring + 3.0s sliding-window trigger

## State
Closed. The sandbox worker (attempts 1 and 2) was interrupted (Ctrl+C) after
writing `src/ov1/model/reconstruction_trigger.py` but before adding tests,
wiring exports, or closing the bead. Finished in the interactive session.

## What exists
- `src/ov1/model/reconstruction_trigger.py` (written by the sandbox worker,
  reviewed here, unmodified):
  - `reconstruction_error_score(x, x_hat)` — spec 4.2's `S_r = ||X - X_hat||^2`
    as a literal sum of squares (not a mean), accumulated in float64.
    Raises `ReconstructionScoreError` on shape mismatch / non-finite input.
  - `DrowningTrigger(tau_drowning, min_continuous_s)` — strict continuity
    state machine; any reading at or below tau resets `sustained_s` to zero.
    Both calibration knobs are required arguments with no default.
  - `evaluate_synthetic_traces()` / `self_check()` — synthetic trace replay.
- `tests/test_model_reconstruction_trigger.py` (added here) — 14 tests.
- `src/ov1/model/__init__.py` — exports wired (added here).

## Validation
`uv run python -m unittest discover tests` → **209 tests, OK** (was 195).
`uv run python -m ov1.model.reconstruction_trigger` → self-check OK.

## Acceptance criteria
- Continuity verified by self-check + tests: a 2.9s spike does not fire, the
  same spike past 3.0s does, a single sub-threshold reading restarts the clock.
- `tau_drowning` exposed as a required per-site calibration argument.
- **FNR/FPR against spec 2's <0.01% / <2.0%: NOT measured.** There is no ocean
  footage, trained autoencoder, or held-out baseline corpus in this repo.
  `synthetic_fnr`/`synthetic_fpr` measure the continuity logic against
  hand-built traces only and must not be reported as spec 2 evidence.

## Git
Nothing committed (conservative profile). Uncommitted, spanning several beads:
`src/ov1/data/{__init__,dataset}.py`, `src/ov1/model/__init__.py` (M);
`src/ov1/model/{autoencoder,reconstruction_trigger,rip_current}.py` and six
`tests/test_*.py` (??).
