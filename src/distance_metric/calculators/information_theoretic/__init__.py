"""
Information-theoretic divergences (KL, Jensen–Shannon, Bhattacharyya, Hellinger).

Exports:
--------
Four divergence calculators and InformationTheoreticDistanceMetric enum.
"""

from .calculators import (
    BhattacharyyaDistanceCalculator,
    HellingerDistanceCalculator,
    JensenShannonDivergenceDistanceCalculator,
    KLDivergenceDistanceCalculator,
)
from .metric import InformationTheoreticDistanceMetric

__all__ = [
    "BhattacharyyaDistanceCalculator",
    "HellingerDistanceCalculator",
    "InformationTheoreticDistanceMetric",
    "JensenShannonDivergenceDistanceCalculator",
    "KLDivergenceDistanceCalculator",
]
