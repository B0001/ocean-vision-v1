"""Provenance manifest and baseline/exclusion gating for training clips
(spec 4.2's "trained *exclusively* on baseline, non-distress aquatic motion
patterns", bead ocean-vision-v1-4ci).

The autoencoder this dataset feeds (ocean-vision-v1-133) learns "what normal
water motion reconstructs like"; anything with distress dynamics mixed into
that corpus teaches it to reconstruct distress too, which quietly destroys
the whole anomaly-detection mechanism without ever raising an exception. This
module is the allow-list gate that stands between a raw clip manifest and
training: a clip is included only if it is explicitly, affirmatively labeled
"baseline" by a human reviewer and carries no exclusion reason. Everything
else -- unlabeled, mislabeled, unreviewed, or explicitly flagged -- is
excluded by default, with the reason recorded so the exclusion is auditable
rather than a silent drop.

This module does not touch the clip's pixel data at all -- see `dataset.py`
for the `ClipFrameSource` that reads frames off the path a manifest entry
names.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

#: The one label value that admits a clip to the training corpus. Anything
#: else (misspelled, "distress", "unknown", "unreviewed", absent) is
#: excluded -- an allow-list, not a deny-list, because the failure mode this
#: guards against (silent contamination) is worse than the failure mode a
#: deny-list would guard against (rejecting a clip that was actually fine).
REQUIRED_BASELINE_LABEL = "baseline"

#: The only two split values a gated manifest entry may carry. "train" feeds
#: the autoencoder; "held_out" is reserved for tau_drowning calibration
#: (spec 4.2) and must never appear in a training DataLoader -- calibrating
#: a threshold against clips the model trained on overstates how well it
#: generalizes.
VALID_SPLITS = ("train", "held_out")

_REQUIRED_FIELDS = ("clip_id", "path", "label", "source", "site_id", "capture_date", "reviewed_by", "split")


class ManifestError(ValueError):
    """Raised when a manifest entry is structurally invalid -- a missing
    required field, an empty clip_id, or a split value outside
    `VALID_SPLITS`. Per this repo's fail-loud rule, a malformed manifest row
    is a curation bug that must stop the load, not get skipped or defaulted
    into a guess about which split or label was intended."""


@dataclass(frozen=True)
class ClipProvenance:
    """One training clip's provenance record.

    Required fields (spec: "documented provenance/exclusion criteria per
    clip"):
      - `clip_id`: unique identifier, non-empty.
      - `path`: where the clip's preprocessed frame data lives on disk (see
        `dataset.NpzClipFrameSource`).
      - `label`: the reviewer's classification. Only exactly
        `REQUIRED_BASELINE_LABEL` ("baseline") admits the clip; any other
        value excludes it.
      - `source`: where the footage came from (e.g. "site-A-buoycam-2026Q1",
        "synthetic-baseline-v1") -- lets a reviewer trace a suspect clip back
        to its origin.
      - `site_id`: installation the footage was captured at. Water motion
        statistics are site-specific; mixing sites into one baseline corpus
        without recording which is which makes a later per-site recalibration
        impossible to audit.
      - `capture_date`: ISO 8601 date the footage was captured.
      - `reviewed_by`: the human who assigned `label`. A `label` with no
        reviewer on record is not provenance, it's an assertion -- this
        field exists so "baseline" traces back to an accountable reviewer,
        not just a filename someone trusted.
      - `split`: "train" or "held_out" (`VALID_SPLITS`).

    Optional:
      - `exclusion_reason`: if a human reviewer sets this, the clip is
        excluded regardless of `label` -- e.g. a clip that genuinely is
        baseline motion but has a desynced sensor pair or a compression
        artifact the reviewer doesn't want in the training set. This is
        deliberately independent of `label`: label says what the footage
        shows, `exclusion_reason` says whether to use it anyway.
      - `notes`: free-text reviewer commentary, not used for gating.
    """

    clip_id: str
    path: Path
    label: str
    source: str
    site_id: str
    capture_date: str
    reviewed_by: str
    split: str
    exclusion_reason: Optional[str] = None
    notes: str = ""


@dataclass(frozen=True)
class GatedManifest:
    """The result of applying baseline/exclusion gating to a raw list of
    `ClipProvenance` records.

    `included` is the allow-listed, trainable corpus. `excluded` is every
    other record paired with why it was dropped -- kept, not discarded, so a
    reviewer can audit the gate's decisions rather than trust them blindly.
    """

    included: tuple[ClipProvenance, ...]
    excluded: tuple[tuple[ClipProvenance, str], ...]

    @property
    def included_train(self) -> tuple[ClipProvenance, ...]:
        """Included clips in the "train" split -- what a training
        DataLoader should be built from."""
        return tuple(c for c in self.included if c.split == "train")

    @property
    def included_held_out(self) -> tuple[ClipProvenance, ...]:
        """Included clips in the "held_out" split -- reserved for
        tau_drowning threshold calibration (spec 4.2), never for training."""
        return tuple(c for c in self.included if c.split == "held_out")


def load_manifest(path: Path) -> list[ClipProvenance]:
    """Load a JSONL manifest (one clip record per line) into
    `ClipProvenance` records.

    Raises `ManifestError` on any structurally invalid row: a missing
    required field, an empty `clip_id`, or a `split` outside `VALID_SPLITS`.
    Does not apply baseline/exclusion gating -- call `gate_baseline_clips` on
    the result for that.
    """
    path = Path(path)
    clips: list[ClipProvenance] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            clips.append(_clip_from_record(record, context=f"{path}:{line_no}"))
    return clips


def _clip_from_record(record: dict, *, context: str) -> ClipProvenance:
    missing = [field for field in _REQUIRED_FIELDS if not record.get(field)]
    if missing:
        raise ManifestError(f"{context}: missing required field(s): {', '.join(missing)}")

    split = record["split"]
    if split not in VALID_SPLITS:
        raise ManifestError(f"{context}: split must be one of {VALID_SPLITS}, got {split!r}")

    return ClipProvenance(
        clip_id=str(record["clip_id"]),
        path=Path(record["path"]),
        label=str(record["label"]),
        source=str(record["source"]),
        site_id=str(record["site_id"]),
        capture_date=str(record["capture_date"]),
        reviewed_by=str(record["reviewed_by"]),
        split=split,
        exclusion_reason=record.get("exclusion_reason") or None,
        notes=str(record.get("notes", "")),
    )


def gate_baseline_clips(clips: Sequence[ClipProvenance]) -> GatedManifest:
    """Apply the baseline allow-list gate (module docstring) to a list of
    provenance records.

    A clip is included iff `label == REQUIRED_BASELINE_LABEL` AND
    `exclusion_reason` is unset. Every excluded clip is paired with a
    concrete, human-readable reason -- never dropped silently.
    """
    included: list[ClipProvenance] = []
    excluded: list[tuple[ClipProvenance, str]] = []

    for clip in clips:
        if clip.label != REQUIRED_BASELINE_LABEL:
            excluded.append(
                (clip, f"label {clip.label!r} is not {REQUIRED_BASELINE_LABEL!r}")
            )
        elif clip.exclusion_reason:
            excluded.append((clip, clip.exclusion_reason))
        else:
            included.append(clip)

    return GatedManifest(included=tuple(included), excluded=tuple(excluded))
