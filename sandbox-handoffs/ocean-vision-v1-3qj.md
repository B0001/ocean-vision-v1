# Handoff: ocean-vision-v1-3qj — Lab-space color normalization with a*/b* contrast stretch

Status: **closed** (acceptance criteria met, evidence below).

## What changed

- `src/ov1/preprocess/__init__.py`, `src/ov1/preprocess/lab.py` (new module):
  - `LabNormalizer(clip_limit, tile_grid_size)` converts a `(H, W, 3)` uint8
    BGR frame to CIE Lab via `cv2.cvtColor(..., COLOR_BGR2Lab)`, then applies
    `cv2.createCLAHE` **only** to the a*/b* planes. L is split off and
    returned untouched — never passed through CLAHE — so it stays
    bit-identical to a plain unstretched conversion for the downstream
    optical-flow bead (`ocean-vision-v1-bag`).
  - `LabFrame` dataclass holds the three `(H, W)` uint8 planes.
  - `clip_limit` and `tile_grid_size` are required constructor args with no
    default — per this repo's calibration-knob rule, real water turbidity
    (the η in e^(−η·d), spec 4.1) is a site property, not something to bake
    in. Shape/dtype are validated and raise `ValueError` rather than coerce
    (fail loud, not silent — the repo's stated failure-mode policy).
  - `self_check()` + `if __name__ == "__main__"` block: standalone runnable
    check (`uv run python -m ov1.preprocess.lab`) that builds a synthetic
    narrow-histogram "turbid" frame, normalizes it, and asserts (a) L is
    bit-identical to the unstretched conversion, (b) the a*/b* histogram
    range widens after CLAHE, (c) a*/b* are actually mutated while L is not.
- `tests/test_preprocess_lab.py` (new): unittest wrapper around the same
  `self_check()` plus focused tests for shape emission, dtype/shape
  validation errors, and constructor validation.
- `tests/test_preprocess_no_network.py` (new): per CLAUDE.md, any bead on
  the ingress→inference→alert path must leave a test that fails if that
  path opens a socket. Lab normalization is spec 4.1, squarely on that path,
  and had no such test — added one following the existing pattern in
  `tests/test_ingress_no_network.py` (patches `socket.socket` /
  `socket.create_connection` to raise, runs `normalize()` through it).
- `pyproject.toml`: added `opencv-python-headless>=5.0.0.93` (no GUI/display
  deps needed on an edge node; this is the first bead in the repo that
  needed OpenCV, per the "first bead that needs a dependency adds it"
  convention in `AGENTS.md`/session prompt). `uv.lock` updated accordingly.

## Acceptance criteria — evidence

- **"L,a*,b* planes emitted per frame"** — `LabFrame.l/a/b`, each `(H, W)`
  uint8; `test_emits_l_a_b_planes_matching_frame_shape` checks shape/dtype
  for a non-square 32×48 frame.
- **"a*/b* histograms visibly de-skewed on turbid test footage"** —
  `test_ab_histograms_de_skewed_on_turbid_footage` asserts post-CLAHE std >
  pre-CLAHE std on a synthetic turbid frame. Measured directly:
  ```
  a range 128–130 (std 0.67) -> 135–143 (std 2.67)
  b range 115–117 (std 0.67) -> 120–131 (std 4.37)
  ```
  (from an ad hoc script run during development, reproduced by
  `self_check()`/the test). This is synthetic data standing in for
  attenuated chrominance, not real ocean turbidity — see caveat below.
- **"L bit-identical to the unstretched conversion"** —
  `test_l_bit_identical_to_unstretched_conversion` does
  `np.testing.assert_array_equal(result.l, cv2.cvtColor(frame,
  COLOR_BGR2Lab)[:, :, 0])`. Passes because `l_plane` is never routed
  through CLAHE in the implementation, not because of a tolerance check.
- **"runnable self-check asserting channel-selective mutation"** —
  `src/ov1/preprocess/lab.py:self_check()`, runnable standalone via
  `uv run python -m ov1.preprocess.lab` (prints `lab_normalize self-check:
  OK`) and exercised by `test_self_check_passes` /
  `test_mutation_is_channel_selective` in the suite.

## Test run (verbatim)

```
$ uv run python -m unittest discover tests
...........................
----------------------------------------------------------------------
Ran 27 tests in 0.108s

OK
```

(17 pre-existing ingress tests + 9 new tests in `test_preprocess_lab.py` + 1
new test in `test_preprocess_no_network.py` = 27. Full breakdown verified
with `-v` during development; the run above is the authoritative pass/fail
count.)

```
$ uv run python -m ov1.preprocess.lab
lab_normalize self-check: OK
```

(A benign `RuntimeWarning` about `ov1.preprocess.lab` being present in
`sys.modules` prints alongside this — expected when running a submodule
with `-m` while its parent package's `__init__.py` also imports it, same
pattern already used by the `ov1.ingress` package. Cosmetic only; the
self-check's own assertions still ran and passed.)

## Constants introduced

| Name | Value | Unit | Source |
|---|---|---|---|
| `clip_limit`, `tile_grid_size` | none — required constructor args | CLAHE clip limit (dimensionless), tile grid (cells) | Deliberately *not* a constant. Bead's own design note says these are a "runtime-tunable calibration knob per deployment site." No default is provided anywhere in `LabNormalizer`. |
| `clip_limit=2.0, tile_grid_size=(8, 8)` | as stated | — | **Guess**, used only inside `self_check()`/tests to exercise the code path on a 64×64 synthetic frame. Not exposed as a library default, not a claim about any real site's calibration. |

No spec numeric target (η, or any of it) is hardcoded into the library code.

## Spec performance targets touched

None of FNR/FPR/inference-latency are touched by this bead — it's a
preprocessing transform, not detection. Spec 4.1's own claim ("mathematically
compensate for depth-dependent light absorption") is not measured here
either: the test only shows CLAHE widens a synthetic narrow histogram, which
says nothing about whether that compensation is *correct* against real
attenuation physics on real footage. That requires real turbid-water clips
and is out of scope for this container.

## What I decided not to do, and why

- **Did not implement DoLP / polarization glare suppression** (also spec
  4.1) — out of scope for this bead, which is Lab normalization only. Not
  filed as a new bead since it's very likely already tracked (spec 4.1 is a
  multi-part section); left for a human to check `bd list` rather than risk
  a duplicate.
- **Did not pick a channel order assumption beyond what the bead already
  specifies.** The bead's own description names `COLOR_BGR2Lab` explicitly,
  so I followed that literally rather than guessing RGB order from spec
  4.1's "RGB Optical" sensor name. This is flagged in the module docstring:
  if the real driver delivers RGB order, callers must convert before calling
  `normalize()`, or red/blue attenuation compensation silently swaps.
  Existing `Frame`/`FrameSource` types in `ov1.ingress` don't fix a channel
  order either, so this ambiguity predates this bead.
- **Did not wire `LabNormalizer` into the ingress ring-buffer pipeline.**
  The bead's acceptance criteria describe a standalone transform + self
  check, not integration; the two blocked beads
  (`ocean-vision-v1-bag` optical flow, `ocean-vision-v1-3fy` tensor assembly)
  are presumably where L/a*/b* get consumed downstream. Wiring it in now
  would be scope creep into beads not claimed this session.
- **Did not use `opencv-python` (with GUI bindings)** — used
  `opencv-python-headless` since this is a headless edge-node pipeline with
  no display; smaller footprint, same `cv2` API surface used here.

## What I could not verify

- Whether CLAHE (vs. some other stretch) is actually the right choice for
  *real* underwater a*/b* attenuation — the bead's design note calls CLAHE
  "the default stretch unless measurement says otherwise," and nothing in
  this container can take that measurement (no ocean footage, no GPU, no
  camera).
- Any real-world de-skewing behavior on actual turbid water footage; the
  synthetic frame in `self_check()`/tests is a narrow-range placeholder
  built to exercise the code path, not a turbidity model.
- Timing / real-time cost of per-frame CLAHE at 4K (spec 4.1 mentions 4K/60
  capture) — not measured, not measurable on this hardware, and this bead's
  acceptance criteria don't ask for it.

## Suggested next commands (not run — conservative git policy)

```bash
git add src/ov1/preprocess tests/test_preprocess_lab.py tests/test_preprocess_no_network.py pyproject.toml uv.lock
git commit -m "..."
```
(`pyproject.toml`/`uv.lock`/`src/`/`tests/` were already untracked at session
start from prior work in this environment; this bead only added files inside
them plus the `opencv-python-headless` dependency line.)
