"""
Concrete similarity-correlation calculator classes.

Exports:
--------
Cosine, Correlation, Spearman, and Kendall distance calculators.
"""

from .correlation import CorrelationDistanceCalculator
from .cosine import CosineDistanceCalculator
from .kendall import KendallDistanceCalculator
from .spearman import SpearmanDistanceCalculator

__all__ = [
    "CorrelationDistanceCalculator",
    "CosineDistanceCalculator",
    "KendallDistanceCalculator",
    "SpearmanDistanceCalculator",
]
