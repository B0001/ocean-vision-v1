"""Tests for ov1.pipeline.window_assembly (bead ocean-vision-v1-0h8): the
glue from a RingBuffer window of SyncedFramePair through LabNormalizer/DoLP
into InferenceTensorAssembler. All synthetic tensors -- no camera, no ocean
footage; see the handoff for what remains unvalidated.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ov1.ingress.frame import Frame
from ov1.ingress.ring_buffer import RingBuffer
from ov1.ingress.sync import SyncedFramePair
from ov1.pipeline.window_assembly import (
    FrameChannelsPipeline,
    SiteCalibrationConfig,
    SiteConfigError,
    assemble_tensor_from_window,
    split_nir_polarization_planes,
)
from ov1.preprocess.dolp import SpecularGlareMasker
from ov1.preprocess.lab import LabNormalizer
from ov1.tensor.assemble import CHANNEL_DOLP, SPEC_CHANNELS, InferenceTensorAssembler


def _synthetic_rgb(height, width, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _synthetic_nir_planes(height, width, seed, glare_region=None):
    """(H, W, 4) NIR frame data: an unpolarized baseline (all four planes
    equal) with an optional strongly-polarized glare patch, mirroring
    ov1.preprocess.dolp's own synthetic fixture."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(80.0, 120.0, size=(height, width))
    i0 = base.copy()
    i45 = base.copy()
    i90 = base.copy()
    i135 = base.copy()
    if glare_region is not None:
        i0[glare_region] = 200.0
        i45[glare_region] = 110.0
        i90[glare_region] = 20.0
        i135[glare_region] = 110.0
    return np.stack([i0, i45, i90, i135], axis=-1)


def _synced_pair(frame_index, height, width, glare_region=None):
    rgb = Frame(sensor_id="rgb_optical", frame_index=frame_index, timestamp_ns=frame_index, data=_synthetic_rgb(height, width, frame_index))
    nir_data = _synthetic_nir_planes(height, width, frame_index, glare_region=glare_region)
    nir = Frame(sensor_id="nir_850nm", frame_index=frame_index, timestamp_ns=frame_index, data=nir_data)
    return SyncedFramePair(frame_index=frame_index, rgb=rgb, nir=nir, skew_ns=0)


class TestSiteCalibrationConfig(unittest.TestCase):
    def test_loads_a_valid_json_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "site.json"
            path.write_text(json.dumps({"clip_limit": 2.0, "tile_grid_size": [8, 8], "dolp_threshold": 0.5}))
            config = SiteCalibrationConfig.from_json_file(path)

        self.assertEqual(config.clip_limit, 2.0)
        self.assertEqual(config.tile_grid_size, (8, 8))
        self.assertEqual(config.dolp_threshold, 0.5)

    def test_missing_field_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "site.json"
            path.write_text(json.dumps({"clip_limit": 2.0, "tile_grid_size": [8, 8]}))
            with self.assertRaises(SiteConfigError):
                SiteCalibrationConfig.from_json_file(path)

    def test_missing_file_raises(self):
        with self.assertRaises(SiteConfigError):
            SiteCalibrationConfig.from_json_file("/nonexistent/path/site.json")

    def test_invalid_json_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "site.json"
            path.write_text("{not valid json")
            with self.assertRaises(SiteConfigError):
                SiteCalibrationConfig.from_json_file(path)

    def test_builds_collaborators(self):
        config = SiteCalibrationConfig(clip_limit=2.0, tile_grid_size=(8, 8), dolp_threshold=0.5)
        self.assertIsInstance(config.build_lab_normalizer(), LabNormalizer)
        self.assertIsInstance(config.build_glare_masker(), SpecularGlareMasker)


class TestSplitNirPolarizationPlanes(unittest.TestCase):
    def test_splits_in_documented_order(self):
        nir_data = _synthetic_nir_planes(8, 8, seed=0)
        i0, i45, i90, i135 = split_nir_polarization_planes(nir_data)
        np.testing.assert_array_equal(i0, nir_data[..., 0])
        np.testing.assert_array_equal(i45, nir_data[..., 1])
        np.testing.assert_array_equal(i90, nir_data[..., 2])
        np.testing.assert_array_equal(i135, nir_data[..., 3])

    def test_wrong_last_axis_raises(self):
        with self.assertRaises(ValueError):
            split_nir_polarization_planes(np.zeros((8, 8, 3)))

    def test_wrong_ndim_raises(self):
        with self.assertRaises(ValueError):
            split_nir_polarization_planes(np.zeros((8, 8)))


class TestFrameChannelsPipeline(unittest.TestCase):
    def setUp(self):
        self.config = SiteCalibrationConfig(clip_limit=2.0, tile_grid_size=(8, 8), dolp_threshold=0.5)
        self.pipeline = FrameChannelsPipeline.from_site_config(self.config)

    def test_convert_produces_matching_plane_shapes(self):
        pair = _synced_pair(0, height=32, width=32)
        channels = self.pipeline.convert(pair)

        self.assertEqual(channels.l.shape, (32, 32))
        self.assertEqual(channels.a.shape, (32, 32))
        self.assertEqual(channels.b.shape, (32, 32))
        self.assertEqual(channels.dolp.shape, (32, 32))
        self.assertTrue(np.all(channels.dolp >= 0.0) and np.all(channels.dolp <= 1.0))

    def test_glare_suppresses_l_plane_in_glare_region(self):
        height, width = 32, 32
        y, x = np.mgrid[0:height, 0:width]
        glare_region = (x - 24) ** 2 + (y - 8) ** 2 < 25

        pair = _synced_pair(0, height, width, glare_region=glare_region)
        channels = self.pipeline.convert(pair)

        self.assertTrue(np.all(channels.dolp[glare_region] > 0.7))
        self.assertTrue(np.all(channels.l[glare_region] == 0))

    def test_convert_window_preserves_order_and_length(self):
        pairs = [_synced_pair(i, 16, 16) for i in range(5)]
        channels_list = self.pipeline.convert_window(pairs)
        self.assertEqual(len(channels_list), 5)

    def test_bad_nir_shape_propagates(self):
        rgb = Frame(sensor_id="rgb_optical", frame_index=0, timestamp_ns=0, data=_synthetic_rgb(16, 16, 0))
        nir = Frame(sensor_id="nir_850nm", frame_index=0, timestamp_ns=0, data=np.zeros((16, 16, 3)))
        pair = SyncedFramePair(frame_index=0, rgb=rgb, nir=nir, skew_ns=0)
        with self.assertRaises(ValueError):
            self.pipeline.convert(pair)


class TestAssembleTensorFromWindow(unittest.TestCase):
    def test_ring_buffer_window_assembles_to_spec_shaped_tensor(self):
        window_frames, height, width = 6, 32, 32
        target_height, target_width = 16, 24

        config = SiteCalibrationConfig(clip_limit=2.0, tile_grid_size=(8, 8), dolp_threshold=0.5)
        pipeline = FrameChannelsPipeline.from_site_config(config)
        assembler = InferenceTensorAssembler(
            target_height=target_height, target_width=target_width, window_frames=window_frames
        )

        buf = RingBuffer(capacity=window_frames)
        for i in range(window_frames):
            buf.push(_synced_pair(i, height, width))
        self.assertTrue(buf.is_full)

        result = assemble_tensor_from_window(buf.window(), pipeline, assembler, pin=False)

        self.assertEqual(
            result.tensor.shape, (1, SPEC_CHANNELS, window_frames, target_height, target_width)
        )
        self.assertEqual(result.tensor.dtype, torch.float32)
        self.assertFalse(result.pinned)
        # DoLP channel (native [0,1]) is not rescaled by the uint8->[0,1]
        # division the L/a*/b* channels get (ocean-vision-v1-3hw) -- values
        # stay within [0, 1] either way, exercised here as a sanity bound.
        dolp_channel = result.tensor[0, CHANNEL_DOLP]
        self.assertTrue(torch.all(dolp_channel >= 0.0) and torch.all(dolp_channel <= 1.0))

    def test_short_window_raises_tensor_assembly_error(self):
        from ov1.tensor.assemble import TensorAssemblyError

        config = SiteCalibrationConfig(clip_limit=2.0, tile_grid_size=(8, 8), dolp_threshold=0.5)
        pipeline = FrameChannelsPipeline.from_site_config(config)
        assembler = InferenceTensorAssembler(target_height=16, target_width=16, window_frames=6)

        pairs = [_synced_pair(i, 16, 16) for i in range(3)]
        with self.assertRaises(TensorAssemblyError):
            assemble_tensor_from_window(pairs, pipeline, assembler, pin=False)


if __name__ == "__main__":
    unittest.main()
