"""
Cosine, Pearson, Spearman, and Kendall distances plus shared bases.

Exports:
--------
Concrete calculators, base classes, and SimilarityCorrelationDistanceMetric.
"""

from .bases import (
    CorrelationDistanceCalculatorBase,
    CosineDistanceCalculatorBase,
    SpearmanDistanceCalculatorBase,
)
from .calculators import (
    CorrelationDistanceCalculator,
    CosineDistanceCalculator,
    KendallDistanceCalculator,
    SpearmanDistanceCalculator,
)
from .metric import SimilarityCorrelationDistanceMetric

__all__ = [
    "CorrelationDistanceCalculator",
    "CorrelationDistanceCalculatorBase",
    "CosineDistanceCalculator",
    "CosineDistanceCalculatorBase",
    "KendallDistanceCalculator",
    "SimilarityCorrelationDistanceMetric",
    "SpearmanDistanceCalculator",
    "SpearmanDistanceCalculatorBase",
]
