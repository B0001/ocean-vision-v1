"""Pretrained video transformer backbone for the ViT embedding branch of the
Spatiotemporal Feature Extractor (spec 3's pipeline diagram: "D -->|3D-CNN /
ViT Embeddings| E"; bead ocean-vision-v1-dxj).

This module only loads a vendored checkpoint from local disk
(`local_files_only=True`, spec 5.2's zero-network-at-runtime requirement) and
adapts its 3-channel RGB patch-embedding stem to this pipeline's 4-channel
[L, a*, b*, DoLP] input (spec 4.2). It does not download anything -- that is
`scripts/vendor_backbone.py`'s job, run once at build time, never at runtime.

Checkpoint: MCG-NJU/videomae-base (VideoMAE, Tong et al. 2022,
arXiv:2203.12602), a ViT-Base masked-autoencoder encoder pretrained on
Kinetics-400, vendored at `models/videomae-base/` (see MANIFEST.json there
for the pinned revision, checksums, and license). Only the encoder
(`VideoMAEModel`) is loaded -- the MAE pretraining decoder is discarded, since
this branch needs embeddings, not a pixel-reconstruction head.

Two things about this checkpoint are load-bearing enough to spell out here
rather than leave implicit:

1. **Manual bias remapping is required for a faithful load.** This checkpoint
   was saved under VideoMAE's original attention parameterization -- learned
   `q_bias`/`v_bias` vectors added to the query/value projections, with the
   key projection carrying **no** bias at all (a deliberate BEiT-style
   choice: a constant added to every key already washes out of a softmax
   over Q.K^T, so the original implementation never trained one). The
   `transformers` version this repo pins (`transformers==5.15.0`) implements
   `VideoMAESelfAttention` with ordinary `nn.Linear(..., bias=True)` query/
   key/value projections and does **not** ship a conversion mapping from the
   checkpoint's `q_bias`/`v_bias` keys to its own `query.bias`/`value.bias`
   names. Call `VideoMAEModel.from_pretrained()` directly on this checkpoint
   and it reports `query.bias`/`key.bias`/`value.bias` as MISSING for every
   one of the 12 encoder layers -- meaning every attention block's bias
   terms come out **randomly initialized**, silently, with no error and no
   loud warning beyond a load report most callers never read. That is not a
   usably "pretrained" encoder; it is a pretrained encoder with 36 vectors of
   noise injected into it. `_remap_videomae_state_dict()` below does this
   translation by hand (`q_bias` -> `query.bias`, `v_bias` -> `value.bias`,
   `key.bias` filled with zeros to match the original's bias-free key
   projection) and `load()` asserts zero missing/unexpected keys against the
   encoder afterward, so a future transformers upgrade that silently changes
   this mapping again fails loud here instead of quietly reintroducing random
   weights.
2. **The pretrained filters were learned on true RGB, not Lab.** Channel-
   adapting the stem (below) keeps the pretrained weights for the first three
   input-channel slots and initializes the fourth (DoLP) from their mean, but
   this pipeline feeds those first three slots [L, a*, b*] (spec 4.2), not
   R/G/B. VideoMAE's low-level filters were trained on genuine RGB
   statistics (ImageNet-style channel correlations); L/a*/b* has different
   per-channel statistics and different cross-channel correlations, so
   whatever those filters learned to detect in RGB is not guaranteed to
   transfer. This module makes the shapes line up and preserves the
   pretrained *weights*; it does not claim the resulting embeddings are
   meaningful on Lab+DoLP input without downstream fine-tuning -- that
   validation is out of this bead's scope (see the handoff).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import VideoMAEConfig, VideoMAEModel

#: Where `scripts/vendor_backbone.py` writes the checkpoint this loader reads.
#: A path, not a URL or repo id -- this module never resolves a Hub
#: reference. Override via `VideoTransformerBackbone(model_dir=...)` for
#: tests or alternate installs; the default is this repo's one vendored copy.
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models" / "videomae-base"

#: Spec 4.2's channel order is [L, a*, b*, DoLP] -- 4 channels. Not a
#: per-site calibration knob (the tensor contract is architectural, fixed by
#: `ov1.tensor.assemble.SPEC_CHANNELS`), so it gets a built-in default like
#: that module's own constants do.
TARGET_NUM_CHANNELS = 4

_QBIAS_RE = re.compile(r"^videomae\.(encoder\.layer\.\d+\.attention\.attention)\.q_bias$")
_VBIAS_RE = re.compile(r"^videomae\.(encoder\.layer\.\d+\.attention\.attention)\.v_bias$")
_VIDEOMAE_PREFIX = "videomae."


class BackboneLoadError(RuntimeError):
    """Raised when the vendored checkpoint doesn't load cleanly into
    `VideoMAEModel` -- a missing file, a state-dict key this module's remap
    doesn't account for, or (after remapping) any remaining missing/
    unexpected key against the encoder. Per this repo's fail-loud rule, a
    partially-loaded backbone must never pass silently as "loaded"."""


class StemAdaptationError(RuntimeError):
    """Raised when the patch-embedding stem can't be channel-adapted -- e.g.
    a target channel count that isn't a growth from the checkpoint's
    original channel count, which this module's mean-init strategy doesn't
    support."""


@dataclass(frozen=True)
class EmbeddingShape:
    """The (seq_len, hidden_size) a `VideoTransformerBackbone.embed()` call
    produces for a given (num_frames, height, width) input, derived from the
    checkpoint's own patch/tubelet geometry (spec: "embedding shape
    documented", bead ocean-vision-v1-dxj acceptance criteria).

    `seq_len = (num_frames // tubelet_size) * (height // patch_size) *
    (width // patch_size)` -- one embedding vector per spatiotemporal patch,
    laid out (T', H', W') row-major then flattened, per
    `VideoMAEPatchEmbeddings` (transformers' `modeling_videomae.py`).
    """

    seq_len: int
    hidden_size: int

    def as_tensor_shape(self, batch_size: int = 1) -> tuple[int, int, int]:
        return (batch_size, self.seq_len, self.hidden_size)


def _remap_videomae_state_dict(raw: dict[str, torch.Tensor], num_hidden_layers: int) -> dict[str, torch.Tensor]:
    """Translate a `VideoMAEForPreTraining`-shaped checkpoint's raw state
    dict into the key names `VideoMAEModel` (bare encoder) expects,
    including the `q_bias`/`v_bias` -> `query.bias`/`value.bias` bias
    remapping this module's docstring explains. Drops every `decoder.*` /
    `mask_token` / `encoder_to_decoder.*` key (the MAE pretraining head --
    this branch wants the encoder only) by virtue of only keeping keys under
    the `videomae.` prefix.
    """
    remapped: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if not key.startswith(_VIDEOMAE_PREFIX):
            continue  # decoder.*, mask_token, encoder_to_decoder.* -- pretraining head, discarded
        m = _QBIAS_RE.match(key)
        if m:
            remapped[f"{m.group(1)}.query.bias"] = value
            continue
        m = _VBIAS_RE.match(key)
        if m:
            remapped[f"{m.group(1)}.value.bias"] = value
            continue
        remapped[key[len(_VIDEOMAE_PREFIX):]] = value

    # The original checkpoint's key projection carries no bias at all (this
    # module's docstring, point 1) -- zero-fill rather than leave
    # `VideoMAEModel`'s freshly-constructed key.bias at its random nn.Linear
    # init, which would silently reintroduce the exact defect this remap
    # exists to avoid.
    for i in range(num_hidden_layers):
        query_bias_key = f"encoder.layer.{i}.attention.attention.query.bias"
        key_bias_key = f"encoder.layer.{i}.attention.attention.key.bias"
        if query_bias_key not in remapped:
            raise BackboneLoadError(
                f"expected {query_bias_key!r} in remapped state dict (from checkpoint's q_bias) "
                f"but it was not found -- checkpoint layout does not match this remap's assumptions"
            )
        remapped[key_bias_key] = torch.zeros_like(remapped[query_bias_key])

    return remapped


def _adapt_patch_embedding_channels(model: VideoMAEModel, num_channels: int) -> None:
    """Replace the patch-embedding `Conv3d`'s input-channel count, keeping
    the pretrained RGB weights for the first (original) channels and
    initializing every added channel as the mean of those RGB filters --
    the standard "inflate" heuristic for adapting an RGB-pretrained stem to
    an extra input channel (e.g. RGB->RGBD). This is a guess, not a spec or
    calibration value: nothing in this repo has validated that mean-of-RGB
    is the right initialization for a DoLP channel specifically, only that
    it is a defensible starting point that doesn't discard any pretrained
    weight. Bias is copied unchanged (a `Conv3d` bias is per-output-channel,
    independent of the input-channel count).

    Mutates `model` in place: swaps `patch_embeddings.projection`, and
    updates `patch_embeddings.num_channels` / `model.config.num_channels` to
    match -- `VideoMAEPatchEmbeddings.forward()` validates its input's
    channel count against `self.num_channels`, so leaving that stale would
    make every adapted-model forward pass raise on exactly the 4-channel
    input this adaptation exists to accept.

    Raises `StemAdaptationError` if `num_channels` is not strictly greater
    than the checkpoint's original channel count -- this heuristic only
    supports growing the stem, not shrinking or leaving it unchanged.
    """
    patch_embeddings = model.embeddings.patch_embeddings
    old_conv = patch_embeddings.projection
    old_channels = old_conv.in_channels

    if num_channels <= old_channels:
        raise StemAdaptationError(
            f"num_channels ({num_channels}) must be greater than the checkpoint's "
            f"original channel count ({old_channels}); this adaptation only supports growing the stem"
        )

    new_conv = nn.Conv3d(
        in_channels=num_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight[:, :old_channels] = old_conv.weight
        mean_filter = old_conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight[:, old_channels:] = mean_filter.expand(-1, num_channels - old_channels, -1, -1, -1)
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    patch_embeddings.projection = new_conv
    patch_embeddings.num_channels = num_channels
    model.config.num_channels = num_channels


class VideoTransformerBackbone:
    """Loads the vendored VideoMAE encoder from local disk, channel-adapts
    its stem to `TARGET_NUM_CHANNELS` (4: [L, a*, b*, DoLP], spec 4.2), and
    exposes it for embedding extraction.

    Construction never touches the network: `model_dir` must already contain
    `config.json` and `model.safetensors` (`scripts/vendor_backbone.py`'s
    output). Raises `BackboneLoadError` if those files are missing or the
    checkpoint's keys don't match this module's remap assumptions.
    """

    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR, num_channels: int = TARGET_NUM_CHANNELS):
        model_dir = Path(model_dir)
        config_path = model_dir / "config.json"
        weights_path = model_dir / "model.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            raise BackboneLoadError(
                f"{model_dir} is missing config.json/model.safetensors -- "
                f"run scripts/vendor_backbone.py to vendor the checkpoint first"
            )

        config = VideoMAEConfig.from_pretrained(model_dir, local_files_only=True)
        model = VideoMAEModel(config)

        raw_state_dict = load_file(weights_path)
        remapped = _remap_videomae_state_dict(raw_state_dict, config.num_hidden_layers)
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        if missing or unexpected:
            raise BackboneLoadError(
                f"checkpoint at {model_dir} did not fully match VideoMAEModel after remap: "
                f"missing={missing!r} unexpected={unexpected!r}"
            )

        self._original_num_channels = config.num_channels
        _adapt_patch_embedding_channels(model, num_channels)
        model.eval()

        self._model = model
        self._config = config
        self._num_channels = num_channels

    @property
    def num_channels(self) -> int:
        return self._num_channels

    @property
    def original_num_channels(self) -> int:
        """The checkpoint's native channel count before adaptation (3, RGB)."""
        return self._original_num_channels

    def embedding_shape(self, num_frames: int, height: int, width: int) -> EmbeddingShape:
        """The (seq_len, hidden_size) `embed()` will produce for an input of
        this (num_frames, height, width) -- see `EmbeddingShape`'s
        docstring for the underlying formula. Does not run the model."""
        tubelet_size = self._config.tubelet_size
        patch_h, patch_w = self._config.patch_size, self._config.patch_size
        if num_frames % tubelet_size != 0:
            raise ValueError(f"num_frames ({num_frames}) must be a multiple of tubelet_size ({tubelet_size})")
        if height % patch_h != 0 or width % patch_w != 0:
            raise ValueError(f"height/width ({height}x{width}) must be multiples of patch_size ({patch_h})")
        seq_len = (num_frames // tubelet_size) * (height // patch_h) * (width // patch_w)
        return EmbeddingShape(seq_len=seq_len, hidden_size=self._config.hidden_size)

    def embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run the adapted encoder on a (B, C, T, H, W) input -- this
        pipeline's tensor convention (`ov1.tensor.assemble`), with
        `C == self.num_channels`. Internally permutes to
        `VideoMAEModel`'s own (B, T, C, H, W) convention before calling it;
        this permute is exactly the kind of silent axis-order mismatch this
        module documents explicitly rather than leaving implicit in a
        `.permute()` call a caller has to discover by reading the source.

        Returns the encoder's `last_hidden_state`, shape
        `(B, seq_len, hidden_size)` -- matching `embedding_shape()` for the
        input's (T, H, W). Raises `ValueError` if `pixel_values` isn't
        5-dimensional or its channel dim doesn't match `self.num_channels`
        (fail loud, not a silent reshape/broadcast).
        """
        if pixel_values.ndim != 5:
            raise ValueError(f"expected a 5D (B, C, T, H, W) tensor, got shape {tuple(pixel_values.shape)}")
        batch, channels, frames, height, width = pixel_values.shape
        if channels != self._num_channels:
            raise ValueError(f"expected {self._num_channels} channels, got {channels}")

        btchw = pixel_values.permute(0, 2, 1, 3, 4)  # (B,C,T,H,W) -> (B,T,C,H,W)
        with torch.no_grad():
            output = self._model(btchw)
        return output.last_hidden_state


def self_check() -> EmbeddingShape:
    """Standalone runnable check (also invoked from the unittest suite):
    loads the vendored backbone offline, adapts its stem, and confirms the
    documented embedding-shape formula against the checkpoint's actual
    (patch_size, tubelet_size, hidden_size) geometry. Does not run a forward
    pass -- see `tests/test_backbone_no_network.py` for why (a real
    16x224x224 forward on this checkpoint takes single-digit seconds of CPU
    time, too slow for every test run; that forward is exercised
    interactively, not in the automated suite -- see the handoff for the
    measured wall time and its caveats).

    Raises `AssertionError` on any failure.
    """
    backbone = VideoTransformerBackbone()
    assert backbone.num_channels == TARGET_NUM_CHANNELS
    assert backbone.original_num_channels == 3

    shape = backbone.embedding_shape(num_frames=16, height=224, width=224)
    assert shape == EmbeddingShape(seq_len=1568, hidden_size=768), f"unexpected shape {shape}"

    return shape


if __name__ == "__main__":
    result = self_check()
    print(f"backbone self-check: OK, embedding shape {result.as_tensor_shape()}")
