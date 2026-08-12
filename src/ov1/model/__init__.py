"""Pretrained backbone(s) feeding the Spatiotemporal Feature Extractor's
ViT embedding branch (spec 3)."""

from .backbone import (
    BackboneLoadError,
    EmbeddingShape,
    StemAdaptationError,
    TARGET_NUM_CHANNELS,
    VideoTransformerBackbone,
)

__all__ = [
    "BackboneLoadError",
    "EmbeddingShape",
    "StemAdaptationError",
    "TARGET_NUM_CHANNELS",
    "VideoTransformerBackbone",
]
