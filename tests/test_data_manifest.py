"""Tests for the baseline provenance manifest and allow-list gate (bead
ocean-vision-v1-4ci, spec 4.2). Covers `load_manifest` parsing/validation and
`gate_baseline_clips`'s label/exclusion-reason gate -- the mechanism that
keeps distress footage out of the autoencoder's training corpus.
"""

import json
import tempfile
import unittest
from pathlib import Path

from ov1.data.manifest import (
    REQUIRED_BASELINE_LABEL,
    VALID_SPLITS,
    ClipProvenance,
    GatedManifest,
    ManifestError,
    gate_baseline_clips,
    load_manifest,
)


def _record(**overrides):
    base = {
        "clip_id": "clip-001",
        "path": "clips/clip-001.npz",
        "label": "baseline",
        "source": "synthetic-baseline-v1",
        "site_id": "site-A",
        "capture_date": "2026-01-15",
        "reviewed_by": "alice",
        "split": "train",
    }
    base.update(overrides)
    return base


def _clip(**overrides):
    base = dict(
        clip_id="clip-001",
        path=Path("clips/clip-001.npz"),
        label="baseline",
        source="synthetic-baseline-v1",
        site_id="site-A",
        capture_date="2026-01-15",
        reviewed_by="alice",
        split="train",
    )
    base.update(overrides)
    return ClipProvenance(**base)


class TestLoadManifest(unittest.TestCase):
    def _write(self, lines):
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        for line in lines:
            f.write(line + "\n")
        f.close()
        self.addCleanup(lambda: Path(f.name).unlink(missing_ok=True))
        return Path(f.name)

    def test_loads_valid_records(self):
        path = self._write(
            [
                json.dumps(_record(clip_id="a", split="train")),
                json.dumps(_record(clip_id="b", split="held_out")),
            ]
        )
        clips = load_manifest(path)
        self.assertEqual([c.clip_id for c in clips], ["a", "b"])
        self.assertEqual(clips[0].path, Path("clips/clip-001.npz"))
        self.assertEqual(clips[1].split, "held_out")

    def test_blank_lines_are_skipped(self):
        path = self._write(["", json.dumps(_record()), "   ", ""])
        clips = load_manifest(path)
        self.assertEqual(len(clips), 1)

    def test_optional_fields_default(self):
        path = self._write([json.dumps(_record())])
        clips = load_manifest(path)
        self.assertIsNone(clips[0].exclusion_reason)
        self.assertEqual(clips[0].notes, "")

    def test_exclusion_reason_and_notes_carry_through(self):
        path = self._write(
            [json.dumps(_record(exclusion_reason="desynced sensor pair", notes="reviewed twice"))]
        )
        clips = load_manifest(path)
        self.assertEqual(clips[0].exclusion_reason, "desynced sensor pair")
        self.assertEqual(clips[0].notes, "reviewed twice")

    def test_missing_required_field_raises(self):
        record = _record()
        del record["reviewed_by"]
        path = self._write([json.dumps(record)])
        with self.assertRaises(ManifestError) as ctx:
            load_manifest(path)
        self.assertIn("reviewed_by", str(ctx.exception))

    def test_empty_required_field_raises(self):
        path = self._write([json.dumps(_record(clip_id=""))])
        with self.assertRaises(ManifestError):
            load_manifest(path)

    def test_invalid_split_raises(self):
        path = self._write([json.dumps(_record(split="training"))])
        with self.assertRaises(ManifestError) as ctx:
            load_manifest(path)
        self.assertIn("split", str(ctx.exception))

    def test_malformed_json_raises_with_line_number(self):
        path = self._write(["{not valid json"])
        with self.assertRaises(ManifestError) as ctx:
            load_manifest(path)
        self.assertIn(":1:", str(ctx.exception))

    def test_valid_splits_are_exactly_train_and_held_out(self):
        self.assertEqual(VALID_SPLITS, ("train", "held_out"))


class TestGateBaselineClips(unittest.TestCase):
    def test_baseline_labeled_clip_with_no_exclusion_is_included(self):
        clip = _clip(label="baseline")
        gated = gate_baseline_clips([clip])
        self.assertEqual(gated.included, (clip,))
        self.assertEqual(gated.excluded, ())

    def test_non_baseline_label_is_excluded_with_reason(self):
        clip = _clip(label="distress")
        gated = gate_baseline_clips([clip])
        self.assertEqual(gated.included, ())
        self.assertEqual(len(gated.excluded), 1)
        excluded_clip, reason = gated.excluded[0]
        self.assertIs(excluded_clip, clip)
        self.assertIn("distress", reason)

    def test_unlabeled_and_unreviewed_style_labels_are_excluded(self):
        for label in ("unknown", "unreviewed", "Baseline", "BASELINE", ""):
            with self.subTest(label=label):
                clip = _clip(clip_id=f"clip-{label!r}", label=label)
                gated = gate_baseline_clips([clip])
                self.assertEqual(gated.included, (), f"label {label!r} must not be admitted")

    def test_exclusion_reason_overrides_baseline_label(self):
        clip = _clip(label="baseline", exclusion_reason="compression artifact")
        gated = gate_baseline_clips([clip])
        self.assertEqual(gated.included, ())
        excluded_clip, reason = gated.excluded[0]
        self.assertIs(excluded_clip, clip)
        self.assertEqual(reason, "compression artifact")

    def test_mixed_corpus_partitions_correctly(self):
        good = _clip(clip_id="good", label="baseline")
        bad_label = _clip(clip_id="bad-label", label="distress")
        bad_excluded = _clip(clip_id="bad-excluded", label="baseline", exclusion_reason="bad sensor")
        gated = gate_baseline_clips([good, bad_label, bad_excluded])
        self.assertEqual(gated.included, (good,))
        self.assertEqual({c.clip_id for c, _ in gated.excluded}, {"bad-label", "bad-excluded"})

    def test_every_excluded_clip_has_a_reason(self):
        clips = [
            _clip(clip_id="a", label="distress"),
            _clip(clip_id="b", label="baseline", exclusion_reason="x"),
        ]
        gated = gate_baseline_clips(clips)
        for _, reason in gated.excluded:
            self.assertTrue(reason)

    def test_required_baseline_label_constant(self):
        self.assertEqual(REQUIRED_BASELINE_LABEL, "baseline")


class TestGatedManifestSplits(unittest.TestCase):
    def test_included_train_and_held_out_partition_by_split(self):
        train_clip = _clip(clip_id="t", label="baseline", split="train")
        held_out_clip = _clip(clip_id="h", label="baseline", split="held_out")
        gated = gate_baseline_clips([train_clip, held_out_clip])
        self.assertEqual(gated.included_train, (train_clip,))
        self.assertEqual(gated.included_held_out, (held_out_clip,))

    def test_excluded_clips_never_appear_in_either_split_property(self):
        excluded_but_train_split = _clip(clip_id="e", label="distress", split="train")
        gated = gate_baseline_clips([excluded_but_train_split])
        self.assertEqual(gated.included_train, ())
        self.assertEqual(gated.included_held_out, ())
        self.assertEqual(len(gated.excluded), 1)

    def test_held_out_split_never_leaks_into_included_train(self):
        held_out_clip = _clip(clip_id="h", label="baseline", split="held_out")
        gated = gate_baseline_clips([held_out_clip])
        self.assertNotIn(held_out_clip, gated.included_train)
        self.assertIn(held_out_clip, gated.included_held_out)

    def test_gated_manifest_is_a_frozen_dataclass(self):
        gated = GatedManifest(included=(), excluded=())
        with self.assertRaises(Exception):
            gated.included = (_clip(),)


if __name__ == "__main__":
    unittest.main()
