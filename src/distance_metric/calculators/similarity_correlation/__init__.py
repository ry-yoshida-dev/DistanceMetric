"""
Cosine, Pearson, Spearman, and Kendall distances.

Exports:
--------
Concrete calculators and SimilarityCorrelationDistanceMetric.
"""

from .calculators import (
    CorrelationDistanceCalculator,
    CosineDistanceCalculator,
    KendallDistanceCalculator,
    SpearmanDistanceCalculator,
)
from .metric import SimilarityCorrelationDistanceMetric

__all__ = [
    "CorrelationDistanceCalculator",
    "CosineDistanceCalculator",
    "KendallDistanceCalculator",
    "SimilarityCorrelationDistanceMetric",
    "SpearmanDistanceCalculator",
]
