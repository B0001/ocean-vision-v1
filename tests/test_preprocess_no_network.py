"""Lab normalization and DoLP glare masking sit on the ingress -> inference
path (spec 4.1), which per spec 5.2 must never touch the network. This
patches socket construction to fail loudly and drives both through it, so
any future change here that adds networking (e.g. a remote calibration
fetch) breaks this test.
"""

import socket
import unittest
from unittest import mock

from ov1.preprocess.dolp import SpecularGlareMasker, _synthetic_polarized_frame, compute_dolp, compute_stokes
from ov1.preprocess.lab import LabNormalizer, _synthetic_turbid_frame


class TestLabNormalizerDoesNotTouchTheNetwork(unittest.TestCase):
    def test_normalize_never_constructs_a_socket(self):
        frame = _synthetic_turbid_frame()
        normalizer = LabNormalizer(clip_limit=2.0, tile_grid_size=(8, 8))

        def _forbidden_socket(*args, **kwargs):
            raise AssertionError("lab normalization attempted to open a network socket")

        with mock.patch.object(socket, "socket", side_effect=_forbidden_socket), \
                mock.patch.object(socket, "create_connection", side_effect=_forbidden_socket):
            result = normalizer.normalize(frame)

        self.assertIsNotNone(result)


class TestDolpMaskingDoesNotTouchTheNetwork(unittest.TestCase):
    def test_dolp_and_mask_never_construct_a_socket(self):
        i0, i45, i90, i135, _ = _synthetic_polarized_frame()
        masker = SpecularGlareMasker(dolp_threshold=0.5)

        def _forbidden_socket(*args, **kwargs):
            raise AssertionError("DoLP glare masking attempted to open a network socket")

        with mock.patch.object(socket, "socket", side_effect=_forbidden_socket), \
                mock.patch.object(socket, "create_connection", side_effect=_forbidden_socket):
            stokes = compute_stokes(i0, i45, i90, i135)
            dolp = compute_dolp(stokes)
            mask = masker.mask(dolp)

        self.assertIsNotNone(mask)


if __name__ == "__main__":
    unittest.main()
