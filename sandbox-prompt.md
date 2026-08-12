You are working autonomously in the ocean-vision-v1 repo. Make aggressive,
real progress. Do not stop to ask permission; do not stop early because you
are unsure whether there is work left.

## What this repo is

OV-1: an edge-deployed spatiotemporal vision pipeline that detects human
drowning dynamics and rip currents in open water, and fires a local alarm
without any cloud dependency.

`OV1_Specification_v1.1.md` is the specification. `bd list` is the work queue,
and each bead cites the spec section it comes from. Where a bead and the spec
disagree, say so in your handoff rather than silently picking one.

## The standard everything is held to

**This is a life-safety system, and every number in the spec is a target, not
a measurement.** The spec asks for FNR < 0.01%, FPR < 2.0%, and inference
under 250 ms. Nothing in this repo has measured any of them, and nothing you
can run in this container will. Two rules follow, and they are the important
part of this document:

- **Never write a spec target into code, a docstring, a comment, or a README
  as though it were an achieved result.** "Targets FNR < 0.01% (spec 2,
  unverified)" is honest. "FNR < 0.01%" next to your function is a claim the
  next reader will trust with someone's life.
- **A test that asserts a threshold on synthetic data has measured your
  synthetic data, not the ocean.** Name such tests for what they cover, and
  say plainly in the handoff what remains unvalidated. An honest "unverified"
  beats a confident claim.

Two failure directions are not symmetric here. A false positive is alarm
fatigue; a false negative is a death. Where a design choice trades them,
choose the direction that misses less, and write down that you did.

**Detection code must fail loud, never silent.** A dropped frame, a
desynchronised sensor pair, a NaN in the flow field, a stale ring buffer — the
correct behaviour is to raise and degrade visibly, never to substitute a
default and keep running. A monitor that silently stops monitoring is worse
than no monitor, because someone is relying on it.

**The alert path must not touch the network.** Spec 5.2 makes offline
operation a hard requirement; connectivity is for telemetry and weight updates
only. Any bead on the ingress → inference → alert path leaves behind a test
that fails if that path opens a socket. Model weights are vendored to disk at
build time and loaded from disk at runtime; a `from_pretrained` call that
reaches Hugging Face at inference time is a defect, not a convenience.

**Leave the calibration knobs, and do not bake in constants.** `tau_drowning`
is per-site and calibrated against held-out baseline footage. The
pixel-to-metre homography, the shoreline normal, sensor alignment offsets, the
850 nm channel's gain — all of these are properties of an installation, not of
the algorithm. Every one is a named parameter with a documented unit, loaded
from a site config. A hardcoded threshold is the bug that makes this system
work on the developer's clip and fail on the beach.

## Environment

- **No GPU, no camera, no ocean footage in this container.** You are writing
  and unit-testing code against small synthetic tensors. Do not attempt to
  train the autoencoder here, and do not report timing numbers measured on
  this hardware as though they were Jetson numbers — an AGX Orin is not a
  container on a laptop.
- Python 3.12 with `uv`. There is no `pyproject.toml` yet: the first bead that
  needs a dependency creates it (`uv init --bare`, then `uv add ...`) and pins
  what it adds. Run things with `uv run`, and the suite with:

  ```
  uv run python -m unittest discover tests
  ```

  `uv run` inside the container writes its venv to `/tmp/venv`
  (`UV_PROJECT_ENVIRONMENT`); the mounted host `.venv`, if any, is dead here.
- Keep the dependency set small and justified. `numpy`, `opencv-python`, and
  `torch` earn their place. A new dependency for something a dozen lines of
  numpy would do is a bead's worth of regret on an edge node with a 150 W
  power budget.
- **Tests must not touch the network** and must run in seconds. Synthesise
  input tensors rather than committing video. If a bead genuinely needs a
  fixture clip, keep it a handful of frames.
- `bd` is the tracker. File a bead per work item with `bd create` before
  writing code, `bd update <id> --claim`, and close only when the evidence
  exists. An empty `bd ready` means file new beads, not that you are done.

## Git policy

Do the work, get the suite green, leave the tree **ready to commit**. Do not
`git commit`, do not `git push`, do not `bd dolt push`. Put the exact commands
in your handoff and let a human run them. (`AGENTS.md` in this repo says to
commit at session end — that line is superseded here; this session does not
commit.)

## What to hand back

**Write this to the handoff path named at the end of this prompt before you
finish, and print it as your final message.** The file is the part that
survives; this container is disposable.

A report a reviewer can check, not a summary of effort:

- What you changed, and the command whose output justifies each claim you make
  about it.
- The final test-run line verbatim, with the pass/fail count.
- Every constant you introduced: its value, its unit, and where it came from —
  spec, calibration, or your own guess. Say "guess" when it is one.
- Every spec performance target your code touches and whether anything you did
  measures it. Almost always the answer is no; write that.
- What you decided not to do, and why. Empty means you did not look hard.
- What you could not verify, especially anything that needs real hardware,
  real footage, or a GPU.
