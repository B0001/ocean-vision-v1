"""Baseline-only training corpus: provenance manifest, allow-list gating,
and a spec-4.2-shaped windowed `Dataset` (bead ocean-vision-v1-4ci)."""

from .dataset import (
    BaselineWindowDataset,
    ClipFrameSource,
    ClipFrameSourceError,
    InMemoryClipFrameSource,
    NpzClipFrameSource,
    WindowBatch,
    WindowSample,
    collate_window_samples,
    save_clip_npz,
)
from .manifest import (
    REQUIRED_BASELINE_LABEL,
    VALID_SPLITS,
    ClipProvenance,
    GatedManifest,
    ManifestError,
    gate_baseline_clips,
    load_manifest,
)

__all__ = [
    "BaselineWindowDataset",
    "ClipFrameSource",
    "ClipFrameSourceError",
    "InMemoryClipFrameSource",
    "NpzClipFrameSource",
    "WindowBatch",
    "WindowSample",
    "collate_window_samples",
    "save_clip_npz",
    "REQUIRED_BASELINE_LABEL",
    "VALID_SPLITS",
    "ClipProvenance",
    "GatedManifest",
    "ManifestError",
    "gate_baseline_clips",
    "load_manifest",
]
