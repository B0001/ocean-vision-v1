"""Tests for the baseline-only windowed `Dataset`/`DataLoader` (bead
ocean-vision-v1-4ci, spec 4.2). Runs against small synthetic frames at a
small target resolution -- this validates window indexing, shape, and the
DataLoader collate path, not real footage; see `ov1.tensor.assemble`'s own
module docstring for that caveat, which applies here too since this dataset
assembles through the same `InferenceTensorAssembler`.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ov1.data.dataset import (
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
from ov1.data.manifest import ClipProvenance
from ov1.tensor.assemble import (
    SPEC_CHANNELS,
    SPEC_HEIGHT,
    SPEC_TEMPORAL_WINDOW_FRAMES,
    SPEC_WIDTH,
    FrameChannels,
)


def _clip_provenance(clip_id, split="train", **overrides):
    base = dict(
        clip_id=clip_id,
        path=Path(f"{clip_id}.npz"),
        label="baseline",
        source="synthetic-baseline-v1",
        site_id="site-A",
        capture_date="2026-01-15",
        reviewed_by="alice",
        split=split,
    )
    base.update(overrides)
    return ClipProvenance(**base)


def _synthetic_frames(n_frames, height=6, width=8, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(n_frames):
        frames.append(
            FrameChannels(
                l=rng.integers(0, 256, size=(height, width), dtype=np.uint8),
                a=rng.integers(0, 256, size=(height, width), dtype=np.uint8),
                b=rng.integers(0, 256, size=(height, width), dtype=np.uint8),
                dolp=rng.random((height, width)),
            )
        )
    return frames


class TestInMemoryClipFrameSource(unittest.TestCase):
    def test_len_and_read_window(self):
        frames = _synthetic_frames(10)
        source = InMemoryClipFrameSource(frames)
        self.assertEqual(len(source), 10)
        window = source.read_window(2, 3)
        self.assertEqual(window, frames[2:5])

    def test_out_of_range_window_raises(self):
        source = InMemoryClipFrameSource(_synthetic_frames(5))
        with self.assertRaises(ClipFrameSourceError):
            source.read_window(3, 5)
        with self.assertRaises(ClipFrameSourceError):
            source.read_window(-1, 2)


class TestNpzClipFrameSourceRoundTrip(unittest.TestCase):
    def test_save_and_read_round_trips_exactly(self):
        frames = _synthetic_frames(5, height=4, width=6)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.npz"
            save_clip_npz(path, frames)
            source = NpzClipFrameSource(path)
            self.assertEqual(len(source), 5)
            window = source.read_window(1, 2)
            for got, want in zip(window, frames[1:3]):
                np.testing.assert_array_equal(got.l, want.l)
                np.testing.assert_array_equal(got.a, want.a)
                np.testing.assert_array_equal(got.b, want.b)
                np.testing.assert_array_equal(got.dolp, want.dolp)

    def test_out_of_range_window_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.npz"
            save_clip_npz(path, _synthetic_frames(4))
            source = NpzClipFrameSource(path)
            with self.assertRaises(ClipFrameSourceError):
                source.read_window(0, 5)

    def test_saving_empty_clip_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                save_clip_npz(Path(tmp) / "empty.npz", [])

    def test_missing_file_raises_clip_frame_source_error(self):
        with self.assertRaises(ClipFrameSourceError):
            NpzClipFrameSource(Path("/nonexistent/path/clip.npz"))

    def test_mismatched_plane_shapes_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.npz"
            np.savez(
                path,
                l=np.zeros((3, 4, 4), dtype=np.uint8),
                a=np.zeros((3, 4, 4), dtype=np.uint8),
                b=np.zeros((3, 4, 4), dtype=np.uint8),
                dolp=np.zeros((2, 4, 4)),  # wrong frame count
            )
            with self.assertRaises(ClipFrameSourceError):
                NpzClipFrameSource(path)

    def test_missing_required_array_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.npz"
            np.savez(path, l=np.zeros((3, 4, 4), dtype=np.uint8))
            with self.assertRaises(ClipFrameSourceError):
                NpzClipFrameSource(path)


class _FrameStore:
    """Test helper mapping clip_id -> frame list, used as the
    `frame_source_factory` a `BaselineWindowDataset` calls per clip."""

    def __init__(self, store):
        self._store = store

    def __call__(self, clip: ClipProvenance) -> ClipFrameSource:
        return InMemoryClipFrameSource(self._store[clip.clip_id])


class TestBaselineWindowDatasetShapeAndIndexing(unittest.TestCase):
    def test_yields_correctly_shaped_windows(self):
        height, width, window_frames = 6, 8, 4
        clip = _clip_provenance("c1")
        store = {"c1": _synthetic_frames(window_frames, height, width)}
        ds = BaselineWindowDataset(
            [clip],
            _FrameStore(store),
            window_frames=window_frames,
            target_height=height,
            target_width=width,
        )
        self.assertEqual(len(ds), 1)
        sample = ds[0]
        self.assertIsInstance(sample, WindowSample)
        self.assertEqual(tuple(sample.tensor.shape), (SPEC_CHANNELS, window_frames, height, width))
        self.assertEqual(sample.clip_id, "c1")
        self.assertEqual(sample.start_frame, 0)

    def test_output_shape_property_matches_yielded_tensors(self):
        clip = _clip_provenance("c1")
        store = {"c1": _synthetic_frames(4, 6, 8)}
        ds = BaselineWindowDataset(
            [clip], _FrameStore(store), window_frames=4, target_height=6, target_width=8
        )
        self.assertEqual(ds.output_shape, (SPEC_CHANNELS, 4, 6, 8))
        self.assertEqual(tuple(ds[0].tensor.shape), ds.output_shape)

    def test_defaults_to_spec_4_2_shape(self):
        ds = BaselineWindowDataset([], _FrameStore({}))
        self.assertEqual(
            ds.output_shape, (SPEC_CHANNELS, SPEC_TEMPORAL_WINDOW_FRAMES, SPEC_HEIGHT, SPEC_WIDTH)
        )

    def test_non_overlapping_windows_by_default(self):
        window_frames = 4
        clip = _clip_provenance("c1")
        store = {"c1": _synthetic_frames(window_frames * 3, 4, 4)}
        ds = BaselineWindowDataset(
            [clip], _FrameStore(store), window_frames=window_frames, target_height=4, target_width=4
        )
        self.assertEqual(len(ds), 3)
        starts = sorted(start for _clip_index, start in ds._index)
        self.assertEqual(starts, [0, 4, 8])

    def test_overlapping_stride_produces_more_windows(self):
        window_frames = 4
        clip = _clip_provenance("c1")
        store = {"c1": _synthetic_frames(10, 4, 4)}
        ds = BaselineWindowDataset(
            [clip],
            _FrameStore(store),
            window_frames=window_frames,
            stride=2,
            target_height=4,
            target_width=4,
        )
        # frames [0,10), window 4, stride 2 -> starts 0,2,4,6 (6+4=10 fits)
        self.assertEqual(len(ds), 4)

    def test_short_clip_is_skipped_not_raised(self):
        window_frames = 6
        clip = _clip_provenance("short")
        store = {"short": _synthetic_frames(3, 4, 4)}
        ds = BaselineWindowDataset(
            [clip], _FrameStore(store), window_frames=window_frames, target_height=4, target_width=4
        )
        self.assertEqual(len(ds), 0)
        self.assertEqual(ds.skipped_clips, [clip])

    def test_mixed_short_and_long_clips_only_short_is_skipped(self):
        window_frames = 4
        long_clip = _clip_provenance("long")
        short_clip = _clip_provenance("short")
        store = {
            "long": _synthetic_frames(window_frames, 4, 4),
            "short": _synthetic_frames(2, 4, 4),
        }
        ds = BaselineWindowDataset(
            [long_clip, short_clip],
            _FrameStore(store),
            window_frames=window_frames,
            target_height=4,
            target_width=4,
        )
        self.assertEqual(len(ds), 1)
        self.assertEqual(ds.skipped_clips, [short_clip])
        self.assertEqual(ds[0].clip_id, "long")

    def test_multiple_clips_indexed_correctly(self):
        window_frames = 4
        clips = [_clip_provenance("a"), _clip_provenance("b")]
        store = {
            "a": _synthetic_frames(window_frames, 4, 4, seed=1),
            "b": _synthetic_frames(window_frames, 4, 4, seed=2),
        }
        ds = BaselineWindowDataset(
            clips, _FrameStore(store), window_frames=window_frames, target_height=4, target_width=4
        )
        self.assertEqual(len(ds), 2)
        clip_ids = {ds[i].clip_id for i in range(len(ds))}
        self.assertEqual(clip_ids, {"a", "b"})

    def test_invalid_window_frames_raises(self):
        with self.assertRaises(ValueError):
            BaselineWindowDataset([], _FrameStore({}), window_frames=0)

    def test_invalid_stride_raises(self):
        with self.assertRaises(ValueError):
            BaselineWindowDataset([], _FrameStore({}), stride=0)


class TestDataLoaderIntegration(unittest.TestCase):
    def _make_dataset(self, n_clips=3, window_frames=4, height=4, width=4):
        clips = [_clip_provenance(f"c{i}") for i in range(n_clips)]
        store = {f"c{i}": _synthetic_frames(window_frames, height, width, seed=i) for i in range(n_clips)}
        return BaselineWindowDataset(
            clips, _FrameStore(store), window_frames=window_frames, target_height=height, target_width=width
        )

    def test_default_collate_rejects_window_sample(self):
        # Documents why collate_window_samples exists: plain DataLoader
        # default collate cannot batch a WindowSample dataclass.
        ds = self._make_dataset()
        loader = DataLoader(ds, batch_size=2)
        with self.assertRaises(TypeError):
            next(iter(loader))

    def test_dataloader_with_collate_fn_yields_batched_tensor(self):
        window_frames, height, width = 4, 4, 4
        ds = self._make_dataset(n_clips=3, window_frames=window_frames, height=height, width=width)
        loader = DataLoader(ds, batch_size=2, collate_fn=collate_window_samples, shuffle=False)
        batches = list(loader)
        self.assertEqual(len(batches), 2)  # 3 samples, batch_size=2 -> batches of 2 and 1
        first = batches[0]
        self.assertIsInstance(first, WindowBatch)
        self.assertEqual(tuple(first.tensor.shape), (2, SPEC_CHANNELS, window_frames, height, width))
        self.assertEqual(len(first.clip_ids), 2)
        self.assertEqual(len(first.start_frames), 2)
        self.assertEqual(batches[1].tensor.shape[0], 1)

    def test_collate_empty_batch_raises(self):
        with self.assertRaises(ValueError):
            collate_window_samples([])

    def test_collate_preserves_provenance_alignment(self):
        ds = self._make_dataset(n_clips=2)
        samples = [ds[i] for i in range(len(ds))]
        batch = collate_window_samples(samples)
        for i, sample in enumerate(samples):
            torch.testing.assert_close(batch.tensor[i], sample.tensor)
            self.assertEqual(batch.clip_ids[i], sample.clip_id)
            self.assertEqual(batch.start_frames[i], sample.start_frame)


class TestHeldOutSplitIsolation(unittest.TestCase):
    """spec 4.2 / this bead's acceptance criteria: a held-out split is
    reserved for tau_drowning threshold calibration and must never leak into
    a training DataLoader. BaselineWindowDataset itself is split-agnostic
    (it builds from whatever clip list it's given); the isolation is the
    caller's responsibility via GatedManifest.included_train /
    included_held_out (see test_data_manifest.py) -- this test demonstrates
    the intended end-to-end usage.
    """

    def test_dataset_built_from_train_split_never_yields_held_out_clip_ids(self):
        from ov1.data.manifest import gate_baseline_clips

        window_frames = 4
        train_clip = _clip_provenance("train-1", split="train")
        held_out_clip = _clip_provenance("held-out-1", split="held_out")
        store = {
            "train-1": _synthetic_frames(window_frames, 4, 4, seed=1),
            "held-out-1": _synthetic_frames(window_frames, 4, 4, seed=2),
        }
        gated = gate_baseline_clips([train_clip, held_out_clip])

        train_ds = BaselineWindowDataset(
            gated.included_train,
            _FrameStore(store),
            window_frames=window_frames,
            target_height=4,
            target_width=4,
        )
        clip_ids = {train_ds[i].clip_id for i in range(len(train_ds))}
        self.assertEqual(clip_ids, {"train-1"})
        self.assertNotIn("held-out-1", clip_ids)


if __name__ == "__main__":
    unittest.main()
