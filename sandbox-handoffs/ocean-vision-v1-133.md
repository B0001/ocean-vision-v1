# ocean-vision-v1-133 — Implement PyTorch Spatiotemporal Autoencoder (Head A)

## What I changed

- **`src/ov1/model/autoencoder.py` (new)** — `SpatiotemporalAutoencoder`
  (`nn.Module`): symmetric Conv3D encoder / ConvTranspose3d decoder, 4
  input/output channels (`ov1.tensor.assemble.SPEC_CHANNELS`), spatial
  downsampling before the bottleneck via `kernel=4, stride=2, padding=1` on
  H/W (exact halve/double for even input, verified below); temporal
  dimension preserved by default (`kernel=3, stride=1, padding=1`, exact
  identity), per the bead's design note scoping downsampling to "spatial".
  Also: `train_autoencoder()` (MSE reconstruction training loop, design
  note's "MSE reconstruction objective"), `measure_inference_latency()` +
  `LatencyMeasurement` (a reusable, honestly-labeled timing utility), and
  `self_check()`.
- **`src/ov1/model/__init__.py`** — exported the new module's public names
  alongside the existing backbone/rip_current exports.
- **`tests/test_model_autoencoder.py` (new)** — 16 tests: forward-shape
  invariant (multiple shapes, both the default temporal-preserving mode and
  `temporal_stride=2`), constructor validation, training-loop mechanics on
  synthetic baseline-gated windows, latency-utility behavior, `self_check()`,
  and a no-network guard on the forward/train path.

Command whose output justifies the shape/training/latency claims:

```
$ uv run python -m unittest tests.test_model_autoencoder -v
...
Ran 16 tests in 0.135s
OK
```

Full suite (this bead's tests plus everything already in the tree,
including the uncommitted work from other closed beads described below):

```
$ uv run python -m unittest discover tests
...................................................................................................................................................................................................
----------------------------------------------------------------------
Ran 195 tests in 3.866s

OK
```
(179 tests existed before this session; 195 - 179 = 16, exactly this
session's new file, and none of the prior 179 regressed.)

Manual acceptance-criterion-1 demonstration (forward pass returns a tensor
matching input shape) at a resolution that shares 1080x1920's
divisible-by-8 property, scaled down to something this container can run in
milliseconds:

```
$ uv run python -c "
import torch
from ov1.model.autoencoder import SpatiotemporalAutoencoder, measure_inference_latency
model = SpatiotemporalAutoencoder()
x = torch.rand(1, 4, 8, 40, 64)
y = model(x)
print('forward shape match:', tuple(x.shape) == tuple(y.shape))
print(measure_inference_latency(model, x, num_iterations=5))
"
forward shape match: True
LatencyMeasurement(mean_ms=8.62, min_ms=6.59, max_ms=15.22, num_iterations=5, device='cpu')
```
That 8.62ms is a small-tensor, no-GPU, this-container number — see "Spec
targets" below for why it says nothing about the spec-2 250ms target.

## Every constant I introduced

| Name | Value | Unit | Source |
|---|---|---|---|
| `DEFAULT_BASE_CHANNELS` | 16 | channels | Guess — architecture width, not measured or tuned. |
| `DEFAULT_DEPTH` | 3 | encoder/decoder block pairs | Derived from spec 4.2's own 1080x1920 resolution: `1080 = 2**3 * 135`, `1920 = 2**7 * 15` — H is the binding constraint, so depth 3 is the largest depth at which spec's own working resolution halves evenly at every block. Not a per-site knob (it's a property of the spec's fixed working resolution), not a performance measurement. |
| `DEFAULT_NORM_GROUPS` | 8 | GroupNorm groups | Guess. Chose GroupNorm over BatchNorm3d because the real inference path always runs batch=1 (`InferenceTensorAssembler.output_shape`'s batch dim) and training batches from a small edge baseline corpus may also be small — that choice (GroupNorm vs BatchNorm) is a documented decision, but 8 as the specific group count is unvalidated. |
| `_LEAKY_RELU_SLOPE` | 0.1 | — | Guess — standard LeakyReLU default for this kind of stack. |
| `DEFAULT_LR` | 1e-3 | Adam learning rate | Guess — common Adam starting point, unvalidated against this model or any real footage. |
| Conv/ConvTranspose H/W `kernel=4, stride=2, padding=1` | — | — | Design choice, not a guess in the "unvalidated number" sense: this specific (kernel, stride, padding) triple is the one for which `Conv3d` halves an even input exactly and `ConvTranspose3d` doubles it back exactly (arithmetic shown in the module docstring and verified by `test_forward_returns_tensor_matching_input_shape`), which is what lets `forward()` reconstruct the exact input shape without ever cropping/padding. |
| T-axis `kernel=3, stride=1, padding=1` (default `temporal_stride=1`) | — | — | Same reasoning, T-axis identity case: exact `out = in`. Chosen because the bead's own design note scopes "downsampling ... required to fit the power envelope" to spatial dims only; nothing here measured whether temporal downsampling is safe for the anomaly signal, so the default leaves it off. `temporal_stride=2` exists as an opt-in for a future bead to explore, using the same exact-halving construction as H/W. |
| Sigmoid on the decoder's final layer | — | — | Not a guess — bounds reconstruction to `[0, 1]`, which is `ov1.tensor.assemble`'s own documented output contract (bead ocean-vision-v1-3hw: "all four channels of the returned tensor are in [0, 1]"). |

None of these are `tau_drowning`-style per-site calibration knobs; they are
model-architecture hyperparameters, fixed at training time, which is why
they get built-in defaults rather than being routed through a site config
— consistent with this repo's stated distinction between "properties of an
installation" (must be a site-config parameter) and "properties of the
algorithm" (may have a default).

## Spec performance targets touched, and what (if anything) measures them

- **Forward pass returns a tensor matching input shape** (bead acceptance
  criterion): measured, by the tests above, on synthetic tensors including
  a 40x64x8-frame case sharing spec 1080x1920's divisibility-by-8 property.
  Not measured at the literal `(1,4,120,1080,1920)` shape — that tensor is
  ~1GB of float32 by itself before any intermediate activations, and this
  container has no GPU; running it here would prove nothing about the
  Jetson target beyond "it didn't crash slowly."
- **Training loop converges on baseline clips** (bead acceptance
  criterion): `test_loss_decreases_on_synthetic_baseline_windows` shows MSE
  loss decreasing over 10 epochs on synthetic random tensors routed through
  the real gated `BaselineWindowDataset` → `DataLoader` →
  `collate_window_samples` path with `label="baseline"` provenance. This
  proves the training loop's *mechanics* are correct (gradients flow,
  optimizer steps reduce loss, the gated-data plumbing fits together) — it
  says nothing about whether the model would converge on, or usefully
  distinguish, real aquatic motion. No real footage exists in this
  container to test that; unverified.
- **Spec 2's inference latency <250ms**: NOT measured. `measure_inference_latency()`
  is a reusable timing utility exercising the real forward-pass code path,
  but every number it can produce in this container is a CPU, no-GPU,
  small-tensor number — not a Jetson AGX Orin number at the real
  `(1,4,120,1080,1920)` shape. Explicitly unverified, and the module's own
  docstrings say so at both the function and dataclass level so a future
  caller can't mistake a container number for a hardware one.
- **Spec 2's FNR < 0.01%, FPR < 2.0%**: NOT touched by this bead at all —
  this module produces reconstructions (`X_hat`), not detections. The
  reconstruction-error scoring and threshold trigger that FNR/FPR would be
  measured against is ocean-vision-v1-wtg, which is blocked on this bead
  and out of scope here.

## What I decided not to do, and why

- **No early stopping, LR scheduling, or checkpointing in `train_autoencoder`.**
  The acceptance criterion is "training loop converges" (loop mechanics),
  not "production training script." Adding those without any real baseline
  corpus or convergence curve to tune them against would be unvalidated
  guesswork layered on top of already-unvalidated guesswork; better to add
  them when there's real data to justify the choices.
- **No temporal downsampling by default.** The design note scopes
  downsampling to "spatial", so I left T untouched by default and exposed
  `temporal_stride=2` as an opt-in rather than silently deciding the
  power-budget tradeoff myself. Flagged in the module docstring as an
  explicit open question for whoever validates this on real hardware:
  120 frames at even a modestly downsampled spatial resolution is still a
  large bottleneck tensor, and nothing in this container can tell you
  whether that fits the Jetson's memory/power budget.
- **Did not implement `ov1.model.rip_current`, `WindowBatch`/`collate_window_samples`,
  or any dataset changes.** Those already existed, uncommitted, in the
  working tree when I started (from prior sessions' work on
  ocean-vision-v1-xe3 and, it appears, unrecorded prep work for this exact
  bead — `WindowBatch`/`collate_window_samples`/`BaselineWindowDataset.output_shape`
  in `src/ov1/data/dataset.py` were present and passing 179/179 tests before
  I touched anything, with no corresponding bead history or handoff for who
  added them). I verified they work (full suite passes) and built on top of
  them (`train_autoencoder` consumes `WindowBatch`), but did not audit or
  modify them — out of this bead's scope.
- **Did not build a script/entrypoint to run training against a real
  manifest.** No real baseline corpus exists in this container to run one
  against; `train_autoencoder` is a library function a future
  training-orchestration bead can call once real data exists.

## What I could not verify

- **Reconstruction quality on real baseline vs. distress footage** — the
  entire point of spec 4.2's design (distress dynamics reconstruct poorly
  because the model only saw baseline motion). No footage of any kind
  exists in this container. This is the single most important unverified
  thing about this bead: the architecture is untested against the actual
  phenomenon it exists to detect.
- **Inference latency against spec 2's <250ms target on the Jetson AGX
  Orin** — no such hardware in this container; see above.
- **Memory/power footprint at the full `(1,4,120,1080,1920)` shape** — not
  run here (would be slow and prove nothing about Jetson memory behavior on
  this container's CPU-only, unrelated memory hierarchy).
- **Whether `DEFAULT_DEPTH=3`/`DEFAULT_BASE_CHANNELS=16`/`DEFAULT_NORM_GROUPS=8`
  are good architecture choices** — no training run against real data
  exists to validate them; they are defensible, documented guesses, not
  measurements.

## Bead status

Claimed and closing as done: forward-pass shape invariant is implemented
and tested; training loop is implemented and its convergence mechanics are
tested on synthetic baseline-gated data; inference-latency measurement is
implemented as a reusable utility with its unverified-on-target-hardware
status documented at every level (module docstring, dataclass docstring,
test comment) rather than silently claimed. The third acceptance-criterion
clause ("inference latency measured against the 250ms target on target
hardware") cannot be satisfied in this container by construction — this
matches this repo's standing rule that "nothing in this repo has measured
[the spec targets], and nothing you can run in this container will."

## Suggested next steps (not done here, out of scope)

- ocean-vision-v1-wtg (already filed, blocked on this bead): reconstruction
  error scoring (`S_r = ||X - X_hat||^2`) and the 3.0s sliding-window
  trigger, consuming this module's `forward()` output.
- Whoever gets Jetson hardware + real footage: run `measure_inference_latency`
  at the real spec shape and an actual `train_autoencoder` run against a
  real gated manifest, then decide whether `temporal_stride=2` is needed
  for the power budget.
