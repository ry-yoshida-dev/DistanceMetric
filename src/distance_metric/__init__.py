"""Pairwise and cross distance / similarity metrics on NumPy arrays."""

from .calculator import DistanceCalculator
from .metric import DistanceMetric
from .result import DistanceResult, DistanceResultType

from .calculators import (
    BinaryDistanceMetric,
    CovarianceNormalizedDistanceMetric,
    GeodesicSequenceDistanceMetric,
    InformationTheoreticDistanceMetric,
    MinkowskiDistanceMetric,
    RatioBasedDistanceMetric,
    SimilarityCorrelationDistanceMetric,
    TransportDistanceMetric,
)

__all__ = [
    "DistanceCalculator",
    "DistanceMetric",
    "DistanceResult",
    "DistanceResultType",
    # metric registry enums (full calculators live under distance_metric.calculators)
    "BinaryDistanceMetric",
    "CovarianceNormalizedDistanceMetric",
    "GeodesicSequenceDistanceMetric",
    "InformationTheoreticDistanceMetric",
    "MinkowskiDistanceMetric",
    "RatioBasedDistanceMetric",
    "SimilarityCorrelationDistanceMetric",
    "TransportDistanceMetric",
]
