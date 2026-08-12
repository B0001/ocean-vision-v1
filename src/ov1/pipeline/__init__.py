"""End-to-end orchestration: wiring the ingress ring buffer's window of
synchronized frame pairs through Lab/DoLP preprocessing into inference
tensor assembly (spec 4.1, 4.2; bead ocean-vision-v1-0h8)."""

from .window_assembly import (
    NIR_PLANE_ORDER,
    FrameChannelsPipeline,
    SiteCalibrationConfig,
    SiteConfigError,
    assemble_tensor_from_window,
    split_nir_polarization_planes,
)

__all__ = [
    "SiteCalibrationConfig",
    "SiteConfigError",
    "FrameChannelsPipeline",
    "assemble_tensor_from_window",
    "split_nir_polarization_planes",
    "NIR_PLANE_ORDER",
]
