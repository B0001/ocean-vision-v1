"""Pretrained backbone(s) feeding the Spatiotemporal Feature Extractor's
ViT embedding branch, and the dual-head inference model's classifiers (spec
3)."""

from .autoencoder import (
    AutoencoderShapeError,
    LatencyMeasurement,
    SpatiotemporalAutoencoder,
    TrainingEpochResult,
    TrainingResult,
    measure_inference_latency,
    train_autoencoder,
)
from .backbone import (
    BackboneLoadError,
    EmbeddingShape,
    StemAdaptationError,
    TARGET_NUM_CHANNELS,
    VideoTransformerBackbone,
)
from .reconstruction_trigger import (
    DrowningObservation,
    DrowningTrigger,
    ReconstructionScoreError,
    SyntheticContinuityResult,
    evaluate_synthetic_traces,
    reconstruction_error_score,
)
from .rip_current import (
    RipChannel,
    RipCurrentClassifier,
    RipCurrentObservation,
    ShorelineNormal,
)

__all__ = [
    "AutoencoderShapeError",
    "LatencyMeasurement",
    "SpatiotemporalAutoencoder",
    "TrainingEpochResult",
    "TrainingResult",
    "measure_inference_latency",
    "train_autoencoder",
    "BackboneLoadError",
    "EmbeddingShape",
    "StemAdaptationError",
    "TARGET_NUM_CHANNELS",
    "VideoTransformerBackbone",
    "DrowningObservation",
    "DrowningTrigger",
    "ReconstructionScoreError",
    "SyntheticContinuityResult",
    "evaluate_synthetic_traces",
    "reconstruction_error_score",
    "RipChannel",
    "RipCurrentClassifier",
    "RipCurrentObservation",
    "ShorelineNormal",
]
