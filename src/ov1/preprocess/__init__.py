"""Optical preprocessing: Lab color-space normalization and DoLP specular
glare suppression (spec 4.1)."""

from .dolp import (
    DolpComputationError,
    GlareMask,
    SpecularGlareMasker,
    StokesParameters,
    compute_dolp,
    compute_stokes,
)
from .lab import LabFrame, LabNormalizer

__all__ = [
    "LabFrame",
    "LabNormalizer",
    "DolpComputationError",
    "GlareMask",
    "SpecularGlareMasker",
    "StokesParameters",
    "compute_dolp",
    "compute_stokes",
]
