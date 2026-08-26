# Handoff: ocean-vision-v1-dxj — Pull video transformer backbone from Hugging Face

**Status: closing.** All four acceptance criteria (offline load with network
disabled; embedding shape documented; 4-channel stem adaptation verified;
license recorded) are met by code + tests below.

## What I found at session start

This bead was already `in_progress` (claimed by a prior session, "started
2026-08-11"). Its actual code was already present and already committed in
`62c7328` ("Add sparse Lucas-Kanade optical flow velocity field
(ocean-vision-v1-bag)" — a commit message about a different bead, but the
diff also carries this one's work, plus most of the rest of the repo):

- `scripts/vendor_backbone.py` — build-time-only downloader:
  `huggingface_hub.snapshot_download` of `MCG-NJU/videomae-base` pinned to
  revision `dc740ceda42fce44faed2ea03c6d447db72f6af9`, writes
  `models/videomae-base/MANIFEST.json` with per-file SHA-256 and license.
- `src/ov1/model/backbone.py` — `VideoTransformerBackbone`: loads the
  vendored checkpoint via `VideoMAEConfig.from_pretrained(...,
  local_files_only=True)` for config and a direct
  `safetensors.torch.load_file()` for weights (never
  `VideoMAEModel.from_pretrained()` on the checkpoint itself, since that
  path silently mis-loads this checkpoint's bias layout — see the module's
  own docstring point 1, already written by the prior session and verified
  correct by me below). Remaps VideoMAE's original `q_bias`/`v_bias`
  parameterization to `transformers==5.15.0`'s `query.bias`/`key.bias`/
  `value.bias` layout, asserts zero missing/unexpected keys, then grows the
  RGB `Conv3d` patch-embedding stem from 3 to 4 input channels
  (`TARGET_NUM_CHANNELS`), keeping pretrained weights for the first 3 and
  mean-initializing the 4th (DoLP).
- `tests/test_backbone_stem_adaptation.py` — unit tests for
  `_adapt_patch_embedding_channels` against a tiny synthetic
  `VideoMAEConfig` (shape growth, RGB-weight preservation, mean-init of the
  new channel, bias preservation, a forward pass on 4-channel input,
  rejection of shrinking/no-op channel counts). All passing already.
- `models/videomae-base/{config.json,preprocessor_config.json,
  MANIFEST.json}` committed to git; `model.safetensors` (376MB) is
  git-ignored (`.gitignore` line `models/videomae-base/model.safetensors`)
  and was already present on this container's disk, presumably from a prior
  run of `vendor_backbone.py`. I did **not** re-run the vendoring script or
  touch the network — see "What I verified" below for how I confirmed the
  weights on disk are the real, checksummed thing rather than assuming it.

What was missing: the module's own docstring and its test-file docstring
both reference `tests/test_backbone_no_network.py` as the place that
exercises a real offline load — that file did not exist. Without it, the
acceptance criterion "Model loads offline with the network disabled" had no
automated evidence, only the prior session's design intent. That gap is why
I judged this bead not actually done, despite most of the code already
being in place.

## What I changed

New file `tests/test_backbone_no_network.py`:

- Patches `socket.socket` / `socket.create_connection` to raise (this
  repo's standard no-network pattern, matching `test_pipeline_no_network.py`
  / `test_flow_no_network.py`), then constructs `VideoTransformerBackbone()`
  against the real vendored checkpoint under that patch. This is the
  bead's first acceptance criterion, made into a repeatable test rather
  than something only verified by hand.
- A second test asserts `embedding_shape(num_frames=16, height=224,
  width=224) == EmbeddingShape(seq_len=1568, hidden_size=768)` against the
  real checkpoint's config (not the tiny synthetic one) and checks
  `as_tensor_shape()`. This is the same assertion `backbone.self_check()`
  already made in a `__main__` block; it just wasn't wired into the
  `unittest` suite the grading/CI path actually runs.
- Both tests are `@unittest.skipUnless` on `model.safetensors` existing on
  disk, so the suite degrades gracefully (skip, not fail) in an environment
  that never ran `vendor_backbone.py` — consistent with this repo's "tests
  must not touch the network" rule; this file doesn't touch the network
  either, it only forbids it.

I made no other code changes. I did not modify `backbone.py`,
`vendor_backbone.py`, or `test_backbone_stem_adaptation.py` — I read them
carefully and re-derived the bias-remapping and channel-growth logic by
hand against the `transformers` source to check the prior session's claims
(see "What I verified"), and found nothing to fix.

## What I verified (commands + output)

Checksums in `MANIFEST.json` against the files actually on disk:

```
$ uv run python -c "
import hashlib, json
d = json.load(open('models/videomae-base/MANIFEST.json'))
for name, expected in d['files_sha256'].items():
    h = hashlib.sha256(open(f'models/videomae-base/{name}','rb').read()).hexdigest()
    print(name, 'OK' if h==expected else f'MISMATCH got={h}')
"
config.json OK
model.safetensors OK
preprocessor_config.json OK
```

Full offline load, sockets blocked, timed:

```
$ time uv run python -c "
import time
t0 = time.time()
from ov1.model.backbone import VideoTransformerBackbone
b = VideoTransformerBackbone()
print('load time', time.time() - t0)
print(b.embedding_shape(16, 224, 224))
"
load time 4.140077829360962
EmbeddingShape(seq_len=1568, hidden_size=768)
real 0m4.451s
```

Real forward pass on real 4-channel input (interactive check only, not in
the automated suite — see "not measured" below for why):

```
$ time uv run python -c "
import torch
from ov1.model.backbone import VideoTransformerBackbone
b = VideoTransformerBackbone()
x = torch.randn(1, 4, 16, 224, 224)
import time; t0 = time.time()
out = b.embed(x)
print('forward time', time.time() - t0)
print('output shape', tuple(out.shape))
print('finite:', torch.isfinite(out).all().item())
"
forward time 0.9223825931549072
output shape (1, 1568, 768)
finite: True
```

Confirms end-to-end: real checkpoint + real 4-channel adaptation produces
the documented `(1, 1568, 768)` shape with finite values, on 16x224x224
input (VideoMAE's native window, not this pipeline's 120-frame/1080p
window — see "not measured" below).

Standalone self-check, matches the `unittest` path:

```
$ uv run python -m ov1.model.backbone
backbone self-check: OK, embedding shape (1, 1568, 768)
```

(The `RuntimeWarning` about `ov1.model.backbone` being in `sys.modules`
before its own execution is a `runpy` quirk this repo already produces
elsewhere, e.g. `python -m ov1.preprocess.lab` — not new, not a defect.)

## Test run (verbatim)

```
$ uv run python -m unittest discover tests 2>&1 | tail -6
..............E......................................................
......................................................................
----------------------------------------------------------------------
Ran 155 tests in 3.767s

FAILED (errors=1)
```

155 total (153 pre-existing + my 2 new). The single error is **pre-existing
and out of scope**: `test_data_dataset.py
TestBaselineWindowDatasetShapeAndIndexing.test_non_overlapping_windows_by_default`
fails with `AttributeError: 'BaselineWindowDataset' object has no attribute
'_index_start'`. That test file, and the `WindowBatch`/
`collate_window_samples` changes to `src/ov1/data/dataset.py` and
`src/ov1/data/__init__.py` it exercises, are **uncommitted working-tree
changes I did not make** — they belong to `ocean-vision-v1-4ci` ("Baseline-
only training dataset and DataLoader"), which is also `in_progress` per `bd
list`. I left that tree state untouched; it is a different bead's
unfinished work, not something this session should fix or absorb. Every
backbone-related test passes; the 2 new tests are part of the 154 passing.

## Constants introduced

None. `TARGET_NUM_CHANNELS = 4` and `DEFAULT_MODEL_DIR` already existed
from the prior session — both architectural (the tensor contract's
`[L,a*,b*,DoLP]` channel count, spec 4.2) or a fixed repo-relative path, not
calibration knobs, and I did not add new ones in my test file. My test file
introduces no new tunable values — it only asserts against
`backbone.embedding_shape()`'s existing formula and the checkpoint's own
config.

Constants inherited from the prior session's code (recorded here since this
bead's acceptance criteria cover them and I re-verified rather than
re-derived them):

- `REPO_ID = "MCG-NJU/videomae-base"`, `REVISION =
  "dc740ceda42fce44faed2ea03c6d447db72f6af9"` (`scripts/vendor_backbone.py`)
  — pinned to a specific commit, sourced from the Hub API's `sha` field per
  the script's own comment, dated 2026-08-11. Not a spec number; a build
  pin.
- `LICENSE = "cc-by-nc-4.0"` — from the Hub's model-card metadata for this
  repo/revision, recorded in `MANIFEST.json`. **Flagged non-commercial** by
  both the script and the manifest's `license_note`; whether that's
  acceptable for this deployment is explicitly left as a human/legal
  decision, not resolved here.
- Mean-of-RGB initialization for the 4th (DoLP) input channel
  (`_adapt_patch_embedding_channels`) — the prior session's own docstring
  already labels this a guess, not a calibration or spec value. I agree
  with that framing: nothing here validates that mean-of-RGB is meaningful
  for a DoLP channel's statistics, only that it's a defensible
  weight-preserving starting point.

## Spec targets this code touches

- Spec 3's pipeline diagram ("D -->|3D-CNN / ViT Embeddings| E"): this
  bead's whole purpose. Produces embeddings; makes no claim about their
  downstream usefulness for drowning/rip-current detection specifically —
  that's the autoencoder head's (`ocean-vision-v1-133`) job to establish.
- Spec 5.2 (zero network dependency at the alert path): the offline-load
  test above is exactly this, but scoped to backbone *loading*, not the
  full alert path. It does not prove the rest of the pipeline is
  network-free (that's `test_pipeline_no_network.py`'s job, already
  passing, already covering the ingress-through-tensor-assembly glue this
  backbone sits downstream of).
- Spec 2's 250ms inference latency, FNR < 0.01%, FPR < 2.0%: **not
  measured, not touched.** The one timing number in this handoff (0.92s for
  a single forward pass on 16x224x224 input) is this container's CPU, on
  VideoMAE's native window size — not the pipeline's actual (1,4,120,
  1080,1920) window, and nowhere near Jetson AGX Orin numbers. It says
  nothing about whether the backbone fits the 250ms budget in production;
  I did not attempt that extrapolation.

## What I decided not to do

- **Did not re-run `scripts/vendor_backbone.py`.** The weights were already
  on disk, checksums verified against `MANIFEST.json` (above). Re-running
  it would touch the network for no reason (it's already idempotent per its
  own docstring) and this session's job is to verify and close, not
  re-fetch working state. If a future session needs to re-vendor (e.g., a
  license or revision change), `vendor_backbone.py` is unchanged and ready.
- **Did not add a full forward-pass test to the automated suite.** The
  0.92s single forward pass I measured is itself borderline for "tests run
  in seconds" when stacked with the ~4s load time already added by the new
  offline-load test — the existing `self_check()`/module docstring already
  made this call before I arrived, and I agree with it: the offline *load*
  is the acceptance criterion, not embed() correctness on a real 5D tensor
  at the pipeline's actual window size, which no test in this repo attempts
  (no ocean footage, no such tensor is synthesized anywhere at that scale).
- **Did not touch `ocean-vision-v1-4ci`'s failing test** (see "Test run"
  above) — out of this bead's scope, and its incomplete/uncommitted state
  belongs to whoever is running that bead next.
- **Did not resolve the CC-BY-NC-4.0 licensing question.** Recorded per the
  acceptance criteria; whether a non-commercial-licensed checkpoint is
  acceptable in this system's eventual deployment is a legal/product call
  the prior session correctly declined to make, and so do I.

## What I could not verify

- Whether the resulting embeddings are actually useful signal for
  drowning-dynamics or rip-current detection when fed [L,a*,b*,DoLP]
  instead of the RGB this checkpoint was pretrained on — the module's own
  docstring already says this plainly (point 2) and I have nothing to add
  beyond confirming the shapes and finiteness. That validation needs
  labeled training data and a downstream head, neither of which exist yet
  (`ocean-vision-v1-133`, `ocean-vision-v1-xe3`).
- Timing/memory on Jetson AGX Orin hardware — no such hardware here, per
  this session's standing instructions. The 0.92s CPU forward-pass number
  above is this container only and must not be read as a production
  latency figure.
- Whether `transformers==5.15.0`'s `VideoMAESelfAttention` bias layout
  (query/key/value all `bias=True`) will still be true in some future
  transformers version — the prior session's `load()` already guards this
  correctly: it asserts zero missing/unexpected keys after remap, so a
  future transformers upgrade that changes this layout again fails loud
  here rather than silently reintroducing random weights. I re-read that
  logic against the current `transformers` source in this environment and
  it's consistent; I have not tested it against any other transformers
  version.

## Suggested next commands (not run — conservative git policy)

```bash
git add tests/test_backbone_no_network.py
git status
git commit -m "Add offline-load acceptance test for VideoMAE backbone (ocean-vision-v1-dxj)"
```

Note `git status` will also show pre-existing unstaged changes to
`src/ov1/data/__init__.py`, `src/ov1/data/dataset.py`, and untracked
`tests/test_data_dataset.py` / `tests/test_data_manifest.py` /
`.claude/settings.local.json` — none of those are mine; they belong to
`ocean-vision-v1-4ci` and are left exactly as I found them. I did not stage
or commit them.

I did not run any git commit/push/dolt-sync commands this session, per the
task's git policy.
