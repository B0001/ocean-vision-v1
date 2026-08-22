"""Conv3D spatiotemporal autoencoder for Head A (spec 4.2, bead
ocean-vision-v1-133): "reconstructs the (1,4,120,1080,1920) input sequence.
Trained exclusively on baseline non-distress aquatic motion so distress
dynamics reconstruct poorly." This is the primary detection mechanism --
`ov1.tensor.assemble.InferenceTensorAssembler` builds the input this module
consumes, `ov1.data.dataset.BaselineWindowDataset` supplies gated baseline
training windows, and the reconstruction-error scoring/trigger that keys off
this module's output is a separate, downstream bead
(ocean-vision-v1-wtg) -- this module produces `X_hat`, it does not compute
`S_r = ||X - X_hat||^2` or decide when to alarm.

Architecture (design note): a symmetric Conv3D encoder / ConvTranspose3d
decoder stack, MSE reconstruction objective, with spatial downsampling before
the bottleneck ("required to fit the 150W/Jetson power envelope"). Every
`Conv3d`/`ConvTranspose3d` in this stack uses `kernel_size=4, stride=2,
padding=1` on the H/W axes -- a choice (not a spec value) picked because it
halves/doubles an even spatial dimension *exactly*
(`out = floor((in + 2*1 - 3 - 1)/2) + 1 = in/2` for even `in`;
`ConvTranspose3d` with the same kernel/stride/padding inverts it exactly:
`out = (in-1)*2 - 2 + 3 + 1 = 2*in`) so `forward()` never needs to crop or pad
the reconstruction to match the input -- a shape mismatch is a real
architecture bug, not an off-by-one this module papers over. Spec 4.2's own
working resolution, 1080x1920, satisfies this exactly up to depth 3
(1080 = 2**3 * 135, 1920 = 2**7 * 15), which is why `DEFAULT_DEPTH = 3`.

Temporal downsampling is off by default (`temporal_stride=1`): the design
note scopes "downsampling ... required to fit the power envelope" to
*spatial* dims, not temporal, so the default architecture leaves T untouched
end to end (kernel=3, stride=1, padding=1 on the T axis is an exact identity:
`out = in`) -- which also means, unlike H/W, no input T needs to be a
particular multiple of anything. `temporal_stride=2` is available for a
future bead that wants a temporally-compressed bottleneck too; this module
does not choose that by default because nothing here has measured whether
120-frame full-temporal-resolution reconstruction is necessary for the
anomaly signal spec 4.2 relies on, or whether it is safe to lose.

Every other architecture number here (channel widths, GroupNorm group count,
LeakyReLU slope, Adam's default `lr`) is a design guess, not a measurement or
a spec value -- called out again at each definition below. None of them are
per-site calibration knobs like `tau_drowning` or the pixel-to-metre
homography; they are properties of the model architecture, fixed at training
time, not of an installation, so unlike those knobs they get built-in
defaults a caller can still override.

What this module does not do, and why:
- It does not compute or threshold reconstruction error (`S_r`) -- that is
  ocean-vision-v1-wtg's job, which blocks on this bead.
- It does not measure inference latency against spec 2's <250ms target, or
  FNR/FPR against spec 2's thresholds -- this container has no GPU, no
  Jetson AGX Orin, and no ocean footage, so nothing here can produce that
  measurement. `measure_inference_latency()` below is a reusable timing
  utility for whoever runs this on real target hardware; its own docstring
  says plainly that a number it returns from this container is not a
  Jetson number.
- Training here (`train_autoencoder`, `self_check`) only demonstrates the
  training loop's *mechanics* converge on synthetic baseline-labeled
  windows -- proof the optimizer step, loss, and gated data path fit
  together, not proof the model learns anything about real water motion.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch import nn

from ov1.data.dataset import WindowBatch
from ov1.tensor.assemble import SPEC_CHANNELS

#: Base width of the first encoder block's output / last decoder block's
#: input; channel width doubles per encoder block after that
#: (`base_channels * 2**i`). A design guess, not a measurement -- nothing in
#: this container has trained this model long enough to know whether a wider
#: or narrower bottleneck reconstructs baseline motion better.
DEFAULT_BASE_CHANNELS = 16

#: Number of downsample/upsample block pairs. 3 is the largest depth for
#: which spec 4.2's 1080x1920 working resolution halves evenly at every
#: block (1080 = 2**3 * 135, 1920 = 2**7 * 15 -- H is the binding
#: constraint); see the module docstring's shape-exactness argument. This is
#: an architectural constant derived from the spec's own resolution, not a
#: per-site knob or a performance measurement.
DEFAULT_DEPTH = 3

#: `nn.GroupNorm`'s group count. GroupNorm (not BatchNorm3d) is used because
#: this pipeline's real inference path runs one window at a time
#: (`InferenceTensorAssembler.output_shape`'s batch dim is always 1) and
#: training batches drawn from a small edge-deployment baseline corpus may
#: also be small -- GroupNorm's statistics don't depend on batch size the
#: way BatchNorm's running stats do. 8 is a common default for this channel
#: range, not a measured choice; `DEFAULT_BASE_CHANNELS` and its doublings
#: are divisible by it by construction (16, 32, 64, ...).
DEFAULT_NORM_GROUPS = 8

#: `nn.LeakyReLU`'s negative slope -- a standard encoder/decoder default
#: (avoids dead units more than plain ReLU on a stack this deep), not tuned
#: against this task.
_LEAKY_RELU_SLOPE = 0.1

#: `torch.optim.Adam`'s default learning rate for `train_autoencoder` --
#: a common starting point, unvalidated against this model or real footage.
DEFAULT_LR = 1e-3


class AutoencoderShapeError(ValueError):
    """Raised when an input's (T, H, W) is incompatible with this model's
    downsample/upsample geometry -- e.g. H or W not divisible by
    `spatial_downsample_factor`. Per this repo's fail-loud rule, a mismatched
    shape must stop the forward pass with a clear cause, not silently
    interpolate/crop the reconstruction into agreement with the input."""


def _spatial_hw_kernel(temporal_stride: int) -> tuple[int, int, int]:
    return (4, 2, 1)  # (kernel, stride, padding) -- always exact halve/double on H/W


def _temporal_kernel(temporal_stride: int) -> tuple[int, int, int]:
    if temporal_stride == 1:
        return (3, 1, 1)  # exact identity on T: out = in
    return (4, 2, 1)  # exact halve/double on T, same construction as H/W


def _encoder_block(in_channels: int, out_channels: int, *, temporal_stride: int, norm_groups: int) -> nn.Module:
    t_kernel, t_stride, t_pad = _temporal_kernel(temporal_stride)
    hw_kernel, hw_stride, hw_pad = _spatial_hw_kernel(temporal_stride)
    return nn.Sequential(
        nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel, hw_kernel, hw_kernel),
            stride=(t_stride, hw_stride, hw_stride),
            padding=(t_pad, hw_pad, hw_pad),
        ),
        nn.GroupNorm(norm_groups, out_channels),
        nn.LeakyReLU(_LEAKY_RELU_SLOPE, inplace=True),
    )


def _decoder_block(
    in_channels: int, out_channels: int, *, temporal_stride: int, norm_groups: int, final: bool
) -> nn.Module:
    t_kernel, t_stride, t_pad = _temporal_kernel(temporal_stride)
    hw_kernel, hw_stride, hw_pad = _spatial_hw_kernel(temporal_stride)
    conv = nn.ConvTranspose3d(
        in_channels,
        out_channels,
        kernel_size=(t_kernel, hw_kernel, hw_kernel),
        stride=(t_stride, hw_stride, hw_stride),
        padding=(t_pad, hw_pad, hw_pad),
    )
    if final:
        # Bounded to [0, 1] with Sigmoid: ov1.tensor.assemble's module
        # docstring documents that "all four channels of the returned
        # tensor are in [0, 1]" (bead ocean-vision-v1-3hw) -- this is that
        # documented input contract, not a guess, so the reconstruction
        # target range is known and bounding the output to match it is a
        # reasonable, cheap inductive bias for the MSE objective.
        return nn.Sequential(conv, nn.Sigmoid())
    return nn.Sequential(conv, nn.GroupNorm(norm_groups, out_channels), nn.LeakyReLU(_LEAKY_RELU_SLOPE, inplace=True))


class SpatiotemporalAutoencoder(nn.Module):
    """Symmetric Conv3D encoder / ConvTranspose3d decoder (design note,
    module docstring). `forward(x)` reconstructs `x`'s exact `(B, C, T, H,
    W)` shape for any input whose H/W are divisible by
    `spatial_downsample_factor` (2**depth) -- spec 4.2's 1080x1920 satisfies
    this at the default depth; small synthetic test shapes must be chosen to
    satisfy it too (`AutoencoderShapeError` otherwise, not a silent
    crop/pad).
    """

    def __init__(
        self,
        in_channels: int = SPEC_CHANNELS,
        base_channels: int = DEFAULT_BASE_CHANNELS,
        depth: int = DEFAULT_DEPTH,
        temporal_stride: int = 1,
        norm_groups: int = DEFAULT_NORM_GROUPS,
    ):
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}")
        if base_channels <= 0:
            raise ValueError(f"base_channels must be > 0, got {base_channels}")
        if depth <= 0:
            raise ValueError(f"depth must be > 0, got {depth}")
        if temporal_stride not in (1, 2):
            raise ValueError(
                f"temporal_stride must be 1 (spatial-only downsampling, this module's default) "
                f"or 2, got {temporal_stride}"
            )
        if norm_groups <= 0:
            raise ValueError(f"norm_groups must be > 0, got {norm_groups}")

        channel_sizes = [base_channels * (2 ** i) for i in range(depth)]
        for i, channels in enumerate(channel_sizes):
            if channels % norm_groups != 0:
                raise ValueError(
                    f"encoder block {i} would have {channels} channels (base_channels={base_channels}, "
                    f"depth={depth}), not divisible by norm_groups={norm_groups}; adjust base_channels "
                    f"or norm_groups so every block's channel count divides evenly"
                )

        encoder_channels = [in_channels] + channel_sizes
        self.encoder = nn.Sequential(
            *[
                _encoder_block(
                    encoder_channels[i], encoder_channels[i + 1], temporal_stride=temporal_stride, norm_groups=norm_groups
                )
                for i in range(depth)
            ]
        )

        decoder_channels = list(reversed(channel_sizes)) + [in_channels]
        self.decoder = nn.Sequential(
            *[
                _decoder_block(
                    decoder_channels[i],
                    decoder_channels[i + 1],
                    temporal_stride=temporal_stride,
                    norm_groups=norm_groups,
                    final=(i == depth - 1),
                )
                for i in range(depth)
            ]
        )

        self.in_channels = in_channels
        self.depth = depth
        self.spatial_downsample_factor = 2 ** depth
        self.temporal_downsample_factor = temporal_stride ** depth

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """The bottleneck latent for an already-validated `(B, C, T, H, W)`
        input. Does not itself validate shape -- call `forward()`, or
        validate via `forward()`'s checks, if `x` isn't already known-good;
        this exists for callers (e.g. a future embedding-inspection tool)
        that only want the encoder half."""
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct `x`. Raises `ValueError` if `x` isn't a 5D tensor
        with `in_channels` channels, or `AutoencoderShapeError` if T/H/W
        aren't compatible with this model's downsample geometry (module
        docstring). Raises `AutoencoderShapeError` (not a silent mismatch)
        if, despite passing those checks, the reconstruction shape still
        doesn't match the input -- this should be unreachable given the
        exact-inversion construction above, and is checked anyway per this
        repo's fail-loud rule rather than trusted blindly.
        """
        self._validate_input(x)
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        if reconstruction.shape != x.shape:
            raise AutoencoderShapeError(
                f"reconstruction shape {tuple(reconstruction.shape)} does not match input shape "
                f"{tuple(x.shape)} -- an architecture bug in the encoder/decoder stride construction, "
                f"not an expected runtime condition"
            )
        return reconstruction

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 5:
            raise ValueError(f"expected a 5D (B, C, T, H, W) tensor, got shape {tuple(x.shape)}")
        _, channels, t, h, w = x.shape
        if channels != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {channels}")
        if h % self.spatial_downsample_factor != 0 or w % self.spatial_downsample_factor != 0:
            raise AutoencoderShapeError(
                f"height/width ({h}x{w}) must be divisible by the encoder's spatial downsample "
                f"factor ({self.spatial_downsample_factor} = 2**depth, depth={self.depth}); "
                f"spec 4.2's 1080x1920 satisfies this at the default depth (1080 = 8*135, 1920 = 8*240)"
            )
        if t % self.temporal_downsample_factor != 0:
            raise AutoencoderShapeError(
                f"temporal length {t} must be divisible by the temporal downsample factor "
                f"({self.temporal_downsample_factor})"
            )


@dataclass(frozen=True)
class TrainingEpochResult:
    """One epoch's mean MSE reconstruction loss across all batches."""

    epoch: int
    mean_loss: float
    num_batches: int


@dataclass(frozen=True)
class TrainingResult:
    """The per-epoch loss trace from one `train_autoencoder` call."""

    epoch_losses: tuple[TrainingEpochResult, ...]

    @property
    def final_loss(self) -> float:
        return self.epoch_losses[-1].mean_loss

    @property
    def initial_loss(self) -> float:
        return self.epoch_losses[0].mean_loss


def train_autoencoder(
    model: SpatiotemporalAutoencoder,
    batches: Iterable[WindowBatch],
    *,
    epochs: int,
    lr: float = DEFAULT_LR,
    device: Optional[torch.device] = None,
) -> TrainingResult:
    """MSE reconstruction training loop (design note: "MSE reconstruction
    objective").

    `batches` must be drawn only from a gated, "train"-split baseline
    corpus -- e.g. `DataLoader(BaselineWindowDataset(gate_baseline_clips(...)
    .included_train, ...), collate_fn=collate_window_samples)`
    (`ov1.data.dataset`). This function does not itself enforce that
    provenance gate; contaminating `batches` with anything else silently
    destroys the anomaly-detection mechanism spec 4.2 relies on
    (`ov1.data.manifest`'s own docstring) -- that gate is the caller's job,
    not re-checked here, since `WindowBatch` carries no provenance/label
    field this function could verify against.

    Does not implement early stopping, learning-rate scheduling, or
    checkpointing -- proving the training loop's mechanics converge is this
    bead's scope; a production training run needs those, but adding them
    without any real baseline corpus or convergence data to tune them
    against would be unvalidated guesswork baked in prematurely.

    `epochs` has no default (a caller must decide it explicitly rather than
    silently getting one epoch); `lr` defaults to `DEFAULT_LR`, itself an
    unvalidated guess (module docstring).

    Raises `ValueError` if `epochs <= 0` or if `batches` is empty (a
    training call over zero data is a caller bug, not something to silently
    report as "trained").
    """
    if epochs <= 0:
        raise ValueError(f"epochs must be > 0, got {epochs}")

    device = device or torch.device("cpu")
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    epoch_results: list[TrainingEpochResult] = []
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        for batch in batches:
            x = batch.tensor.to(device)
            optimizer.zero_grad()
            reconstruction = model(x)
            loss = loss_fn(reconstruction, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        if num_batches == 0:
            raise ValueError("train_autoencoder received zero batches -- empty training set")
        epoch_results.append(TrainingEpochResult(epoch=epoch, mean_loss=total_loss / num_batches, num_batches=num_batches))

    return TrainingResult(epoch_losses=tuple(epoch_results))


@dataclass(frozen=True)
class LatencyMeasurement:
    """Wall-clock forward-pass latency measured **on whatever device this
    process is currently running on**.

    In this repo's dev container that is a CPU with no GPU -- not spec
    5.1's Jetson AGX Orin deployment target, and this is not a measurement
    of spec 2's <250ms inference-latency target. This dataclass and
    `measure_inference_latency()` exist so the exact same timing code runs
    identically on real target hardware; a number returned from this
    container must never be reported as though it were that number (this
    repo's standing rule).
    """

    mean_ms: float
    min_ms: float
    max_ms: float
    num_iterations: int
    device: str


def measure_inference_latency(
    model: SpatiotemporalAutoencoder,
    sample_input: torch.Tensor,
    *,
    num_iterations: int = 5,
    warmup_iterations: int = 1,
) -> LatencyMeasurement:
    """Time `num_iterations` forward passes of `model` on `sample_input`,
    after `warmup_iterations` untimed warmup passes (first-call overhead --
    lazy kernel/allocator init -- would otherwise skew a small sample).
    Runs under `torch.no_grad()` and forces `model.eval()` first, matching
    real inference (not training) conditions.

    Raises `ValueError` if `num_iterations <= 0`. See `LatencyMeasurement`'s
    docstring for why a result from this call is not a spec-2-target
    measurement.
    """
    if num_iterations <= 0:
        raise ValueError(f"num_iterations must be > 0, got {num_iterations}")

    model.eval()
    with torch.no_grad():
        for _ in range(warmup_iterations):
            model(sample_input)

        timings_ms: list[float] = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            model(sample_input)
            timings_ms.append((time.perf_counter() - start) * 1000.0)

    return LatencyMeasurement(
        mean_ms=sum(timings_ms) / len(timings_ms),
        min_ms=min(timings_ms),
        max_ms=max(timings_ms),
        num_iterations=num_iterations,
        device=str(sample_input.device),
    )


def self_check() -> TrainingResult:
    """Standalone runnable check (also invoked from the unittest suite):
    forward pass reconstructs a small synthetic input's exact shape, a few
    training steps over synthetic `WindowBatch`es reduce the MSE loss
    (training-loop *mechanics* only -- module docstring), and the latency
    utility runs without error (this container's CPU, not Jetson).

    Raises `AssertionError` on any failure.
    """
    torch.manual_seed(0)
    model = SpatiotemporalAutoencoder(base_channels=4, depth=2, norm_groups=2)

    batch_size, t, height, width = 1, 4, 16, 16  # H/W divisible by 2**depth=4
    x = torch.rand(batch_size, SPEC_CHANNELS, t, height, width)
    reconstruction = model(x)
    assert reconstruction.shape == x.shape, f"shape mismatch: {tuple(reconstruction.shape)} vs {tuple(x.shape)}"

    synthetic_batches = [
        WindowBatch(tensor=torch.rand(2, SPEC_CHANNELS, t, height, width), clip_ids=["s0", "s1"], start_frames=[0, 0])
        for _ in range(4)
    ]
    training_result = train_autoencoder(model, synthetic_batches, epochs=8, lr=1e-2)
    assert training_result.final_loss < training_result.initial_loss, (
        f"loss did not decrease on synthetic data: {training_result.initial_loss} -> {training_result.final_loss}"
    )

    latency = measure_inference_latency(model, x, num_iterations=3)
    assert latency.mean_ms > 0.0, "measured non-positive latency"
    assert latency.num_iterations == 3

    return training_result


if __name__ == "__main__":
    result = self_check()
    print(
        f"autoencoder self-check: OK, synthetic loss {result.initial_loss:.4f} -> {result.final_loss:.4f}"
    )
