"""Baseline-only training `Dataset`/`DataLoader` for the Head A autoencoder
(spec 4.2, bead ocean-vision-v1-4ci).

Wraps a gated (`manifest.gate_baseline_clips`), "train"-split corpus of
provenance-checked baseline clips and yields spec-shaped
`(C, T, H, W) = (4, 120, 1080, 1920)` reconstruction-training windows,
assembled through the same `InferenceTensorAssembler` the real-time inference
path uses (ocean-vision-v1-3fy) -- training and inference windows are built
by one code path, so a bug in window assembly can't silently diverge between
"what the model saw in training" and "what it sees at alert time".

Video decode (raw ingress frames -> Lab/DoLP planes for an arbitrary stored
clip) is out of scope here -- see ocean-vision-v1-0h8. This module starts
from already-preprocessed `FrameChannels` planes cached to disk one `.npz`
per clip (`NpzClipFrameSource`); producing those files from real footage is
an offline preprocessing step this bead does not build.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ov1.tensor.assemble import (
    SPEC_HEIGHT,
    SPEC_TEMPORAL_WINDOW_FRAMES,
    SPEC_WIDTH,
    FrameChannels,
    InferenceTensorAssembler,
)

from .manifest import ClipProvenance


class ClipFrameSourceError(RuntimeError):
    """Raised when a clip's on-disk frame data doesn't match its own
    internal contract -- e.g. the l/a/b/dolp arrays disagree on frame count
    or spatial shape. A clip that fails this check is corrupt or was
    mis-assembled upstream; per this repo's fail-loud rule, that must stop
    the load, not get truncated or reshaped into agreement."""


class ClipFrameSource(ABC):
    """A single clip's frame data, readable as windows of `FrameChannels`.

    The abstraction that lets `BaselineWindowDataset` stay agnostic of how a
    clip is stored -- `NpzClipFrameSource` reads cached `.npz` planes;
    `InMemoryClipFrameSource` (tests, and any future in-memory pipeline) reads
    a Python list.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Number of frames in this clip."""
        raise NotImplementedError

    @abstractmethod
    def read_window(self, start: int, length: int) -> list[FrameChannels]:
        """Return `length` consecutive frames starting at `start`.

        Implementations must raise (not clamp or pad) if
        `start + length > len(self)` -- an out-of-range window request is a
        caller bug in window-index bookkeeping, not something to silently
        truncate.
        """
        raise NotImplementedError


class InMemoryClipFrameSource(ClipFrameSource):
    """Wraps an in-memory list of `FrameChannels` -- what tests and
    synthetic-corpus tooling use; no filesystem involved."""

    def __init__(self, frames: Sequence[FrameChannels]):
        self._frames = list(frames)

    def __len__(self) -> int:
        return len(self._frames)

    def read_window(self, start: int, length: int) -> list[FrameChannels]:
        if start < 0 or start + length > len(self._frames):
            raise ClipFrameSourceError(
                f"window [{start}, {start + length}) out of range for a {len(self._frames)}-frame clip"
            )
        return self._frames[start : start + length]


class NpzClipFrameSource(ClipFrameSource):
    """Reads a clip's cached `[L, a*, b*, DoLP]` planes from a `.npz` file
    with arrays `l`, `a`, `b` (uint8, `(N, H, W)`) and `dolp` (float,
    `(N, H, W)`) -- the on-disk format `save_clip_npz` writes.

    Loaded with `mmap_mode="r"` so opening a clip doesn't pull its full
    frame sequence into memory before a window is even requested; each
    `read_window` call copies out only the frames it needs.
    """

    def __init__(self, path: Path):
        self._path = Path(path)
        try:
            self._npz = np.load(self._path, mmap_mode="r")
        except (OSError, ValueError) as exc:
            raise ClipFrameSourceError(f"{self._path}: failed to open as clip .npz: {exc}") from exc

        for key in ("l", "a", "b", "dolp"):
            if key not in self._npz:
                raise ClipFrameSourceError(f"{self._path}: missing required array {key!r}")

        shapes = {key: self._npz[key].shape for key in ("l", "a", "b", "dolp")}
        first = next(iter(shapes.values()))
        if any(shape != first for shape in shapes.values()):
            raise ClipFrameSourceError(f"{self._path}: l/a/b/dolp arrays disagree on shape: {shapes}")
        if len(first) != 3:
            raise ClipFrameSourceError(f"{self._path}: expected (N, H, W) arrays, got shape {first}")

        self._length = first[0]

    def __len__(self) -> int:
        return self._length

    def read_window(self, start: int, length: int) -> list[FrameChannels]:
        if start < 0 or start + length > self._length:
            raise ClipFrameSourceError(
                f"{self._path}: window [{start}, {start + length}) out of range for a "
                f"{self._length}-frame clip"
            )
        end = start + length
        l_arr = np.array(self._npz["l"][start:end])
        a_arr = np.array(self._npz["a"][start:end])
        b_arr = np.array(self._npz["b"][start:end])
        dolp_arr = np.array(self._npz["dolp"][start:end])
        return [
            FrameChannels(l=l_arr[i], a=a_arr[i], b=b_arr[i], dolp=dolp_arr[i])
            for i in range(length)
        ]


def save_clip_npz(path: Path, frames: Sequence[FrameChannels]) -> None:
    """Write a clip's `FrameChannels` sequence to the `.npz` layout
    `NpzClipFrameSource` reads. Offline preprocessing/tooling and tests use
    this; the real-time inference path never does (it assembles tensors
    straight from live frames, spec 4.2 / ocean-vision-v1-3fy)."""
    if not frames:
        raise ValueError("cannot save an empty clip (zero frames)")
    np.savez(
        path,
        l=np.stack([f.l for f in frames]),
        a=np.stack([f.a for f in frames]),
        b=np.stack([f.b for f in frames]),
        dolp=np.stack([f.dolp for f in frames]),
    )


@dataclass(frozen=True)
class WindowSample:
    """One training sample: an assembled `(C, T, H, W)` window plus enough
    provenance to trace it back to its source clip -- so a bad reconstruction
    during training debugging can be traced to a specific clip and offset,
    not just an opaque tensor."""

    tensor: torch.Tensor
    clip_id: str
    start_frame: int


@dataclass(frozen=True)
class WindowBatch:
    """A collated batch of `WindowSample`s: a stacked `(B, C, T, H, W)`
    tensor plus per-item provenance, kept as parallel lists rather than
    folded into the tensor so a bad reconstruction during training can still
    be traced to a `(clip_id, start_frame)` after batching."""

    tensor: torch.Tensor
    clip_ids: list[str]
    start_frames: list[int]


def collate_window_samples(samples: Sequence[WindowSample]) -> WindowBatch:
    """`collate_fn` for `DataLoader(BaselineWindowDataset(...), ...)`.

    `WindowSample` is a plain dataclass, not a `Tensor`/`Mapping`/`Sequence`/
    `NamedTuple`, so `torch.utils.data.dataloader.default_collate` raises
    `TypeError` on it -- this is the explicit collate function this dataset
    requires; a `DataLoader` built without it will fail on its first batch.
    Raises `ValueError` on an empty batch rather than returning a
    zero-length tensor a downstream training loop would have to guard
    against.
    """
    if not samples:
        raise ValueError("cannot collate an empty batch of window samples")
    return WindowBatch(
        tensor=torch.stack([sample.tensor for sample in samples], dim=0),
        clip_ids=[sample.clip_id for sample in samples],
        start_frames=[sample.start_frame for sample in samples],
    )


class BaselineWindowDataset(Dataset):
    """A `torch.utils.data.Dataset` over sliding-window baseline clips,
    assembled to the spec 4.2 `(C, T, H, W)` shape.

    Construct from a gated, single-split clip list (typically
    `GatedManifest.included_train` -- see `manifest.py`; passing
    `included_held_out` here would leak calibration data into training,
    which is exactly the split this dataset does not enforce for you, so the
    caller must). `frame_source_factory` maps a `ClipProvenance` to the
    `ClipFrameSource` that reads its frames (`NpzClipFrameSource` in
    production; `InMemoryClipFrameSource` in tests).

    Windows are non-overlapping by default (`stride == window_frames`);
    pass a smaller `stride` for overlapping windows if a future bead wants
    denser sampling from a limited baseline corpus.

    A clip shorter than `window_frames` yields zero windows and is recorded
    in `skipped_clips` rather than raising -- clip length is a fact about the
    corpus, not a system fault, but it must still be visible rather than
    silently dropped: check `skipped_clips` before training and expect it to
    be empty, or know why it isn't.
    """

    def __init__(
        self,
        clips: Sequence[ClipProvenance],
        frame_source_factory: Callable[[ClipProvenance], ClipFrameSource],
        *,
        window_frames: int = SPEC_TEMPORAL_WINDOW_FRAMES,
        stride: Optional[int] = None,
        target_height: int = SPEC_HEIGHT,
        target_width: int = SPEC_WIDTH,
    ):
        if window_frames <= 0:
            raise ValueError(f"window_frames must be > 0, got {window_frames}")
        stride = window_frames if stride is None else stride
        if stride <= 0:
            raise ValueError(f"stride must be > 0, got {stride}")

        self._clips = list(clips)
        self._frame_source_factory = frame_source_factory
        self._window_frames = window_frames
        self._stride = stride
        self._assembler = InferenceTensorAssembler(
            target_height=target_height, target_width=target_width, window_frames=window_frames
        )

        self._index: list[tuple[int, int]] = []  # (clip_index, start_frame)
        self.skipped_clips: list[ClipProvenance] = []

        for clip_index, clip in enumerate(self._clips):
            source = frame_source_factory(clip)
            length = len(source)
            if length < window_frames:
                self.skipped_clips.append(clip)
                continue
            start = 0
            while start + window_frames <= length:
                self._index.append((clip_index, start))
                start += stride

    @property
    def output_shape(self) -> tuple[int, int, int, int]:
        """The `(C, T, H, W)` shape every yielded `WindowSample.tensor`
        has -- the assembler's `(B, C, T, H, W)` `output_shape` with the
        batch dimension dropped, since a `Dataset.__getitem__` yields one
        unbatched window."""
        return self._assembler.output_shape[1:]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> WindowSample:
        clip_index, start = self._index[idx]
        clip = self._clips[clip_index]
        source = self._frame_source_factory(clip)
        frames = source.read_window(start, self._window_frames)

        # pin=False: DataLoader's own pin_memory=True handles pinning on the
        # batched tensor after collation, which is cheaper than pinning
        # every unbatched window individually here.
        assembled = self._assembler.assemble(frames, pin=False)
        return WindowSample(
            tensor=assembled.tensor.squeeze(0),  # (1,C,T,H,W) -> (C,T,H,W)
            clip_id=clip.clip_id,
            start_frame=start,
        )
