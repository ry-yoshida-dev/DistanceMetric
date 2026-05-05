"""
Binary-set distances for aligned vectors.

Exports concrete calculators and BinaryDistanceMetric registry enum.

Exports:
--------
HammingDistanceCalculator, JaccardDistanceCalculator, BinaryDistanceMetric.
Public names are listed in __all__.
"""

from .calculators import HammingDistanceCalculator, JaccardDistanceCalculator
from .metric import BinaryDistanceMetric

__all__ = [
    "HammingDistanceCalculator",
    "JaccardDistanceCalculator",
    "BinaryDistanceMetric",
]
