"""Pixel-to-metre ground-plane homography for one installation (spec 4.3,
bead ocean-vision-v1-bag).

The camera's mounting height, tilt, and focal length -- and therefore the
mapping from a pixel coordinate to a real-world position on the water
surface -- is a property of a specific installation. It is calibrated once
at install time against known reference points (e.g. marked distances on a
dock or a calibration target at measured range) and is never derived or
guessed here. There is no built-in default: callers must supply a matrix
from a site config.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class PointBehindCameraError(RuntimeError):
    """Raised when a pixel maps to a homogeneous world coordinate with a
    near-zero or negative weight -- i.e. the homography places the point at
    or behind the camera's horizon. This is either a bad calibration or a
    tracked point that has left the water plane the homography models
    (e.g. surf line, shoreline, sky). Never silently divided through."""

    def __init__(self, pixel: tuple[float, float], w: float):
        self.pixel = pixel
        self.w = w
        super().__init__(
            f"pixel {pixel} maps to homogeneous weight {w!r}, at/behind the "
            f"calibrated horizon -- point is off the modelled water plane"
        )


@dataclass(frozen=True)
class Homography:
    """A calibrated pixel -> metre ground-plane mapping for one site.

    `matrix` is a (3, 3) float64 array such that for a pixel (x, y), the
    homogeneous world coordinate is `matrix @ [x, y, 1]`, and the metre
    position on the water plane is `(X / w, Y / w)`. This is the standard
    planar homography used to rectify a single camera view of a
    (locally-flat) water surface onto world-plane metres; units of the
    output are whatever units the calibration reference points were
    measured in -- this module assumes metres throughout, per spec 4.3's
    velocity threshold being expressed in m/s.

    No default matrix is provided. A caller passing `np.eye(3)` is passing a
    literal "1 pixel = 1 metre, no perspective" calibration, which is never
    correct for a real installation -- construct this only from a site's own
    calibration procedure.
    """

    matrix: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(f"homography matrix must be (3, 3), got shape {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise ValueError("homography matrix must be finite (no NaN/Inf)")
        if abs(np.linalg.det(matrix)) < 1e-12:
            raise ValueError("homography matrix is singular (determinant ~= 0)")
        object.__setattr__(self, "matrix", matrix)

    def pixel_to_metres(self, points_px: np.ndarray) -> np.ndarray:
        """Map (N, 2) pixel coordinates to (N, 2) metre coordinates on the
        calibrated water plane.

        Raises `PointBehindCameraError` for any point whose homogeneous
        weight is not comfortably positive, rather than returning a divide-
        by-near-zero blowup as a large-but-finite metre position.
        """
        points_px = np.asarray(points_px, dtype=np.float64)
        if points_px.ndim != 2 or points_px.shape[1] != 2:
            raise ValueError(f"expected (N, 2) pixel points, got shape {points_px.shape}")

        homogeneous_px = np.concatenate(
            [points_px, np.ones((points_px.shape[0], 1), dtype=np.float64)], axis=1
        )
        world = homogeneous_px @ self.matrix.T
        w = world[:, 2]

        bad = np.where(w < 1e-9)[0]
        if len(bad) > 0:
            i = int(bad[0])
            raise PointBehindCameraError(pixel=(float(points_px[i, 0]), float(points_px[i, 1])), w=float(w[i]))

        return world[:, :2] / w[:, np.newaxis]
