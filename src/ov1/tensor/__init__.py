"""Inference tensor assembly: stacking per-frame [L, a*, b*, DoLP] planes
into the spec 4.2 (B, C, T, H, W) = (1, 4, 120, 1080, 1920) tensor."""

from .assemble import (
    CHANNEL_A,
    CHANNEL_B,
    CHANNEL_DOLP,
    CHANNEL_L,
    CHANNEL_ORDER,
    SPEC_CHANNELS,
    SPEC_HEIGHT,
    SPEC_TEMPORAL_WINDOW_FRAMES,
    SPEC_WIDTH,
    AssembledTensor,
    FrameChannels,
    InferenceTensorAssembler,
    TensorAssemblyError,
)

__all__ = [
    "AssembledTensor",
    "FrameChannels",
    "InferenceTensorAssembler",
    "TensorAssemblyError",
    "CHANNEL_ORDER",
    "CHANNEL_L",
    "CHANNEL_A",
    "CHANNEL_B",
    "CHANNEL_DOLP",
    "SPEC_CHANNELS",
    "SPEC_TEMPORAL_WINDOW_FRAMES",
    "SPEC_HEIGHT",
    "SPEC_WIDTH",
]
