"""
Similarity and correlation metric names mapped to distance-style scores.

Includes cosine, Pearson, Spearman, and Kendall variants.
"""

from enum import Enum
from typing import Type

from ...calculator import DistanceCalculator
from ...metric import DistanceMetric


class SimilarityCorrelationDistanceMetric(DistanceMetric, Enum):
    """
    Vector similarity metrics mapped to distance-like scores.

    Members:
    --------
    COSINE : str
        Cosine-based distance along the last axis.
    CORRELATION : str
        Pearson correlation turned into a distance after centering.
    SPEARMAN : str
        Correlation distance on ranks per row.
    KENDALL : str
        Distance from pairwise concordance of coordinate ordering.

    Notes:
    -----
    Results remain wrapped as DistanceResult with default distance semantics.
    """

    COSINE = "cosine"
    CORRELATION = "correlation"
    SPEARMAN = "spearman"
    KENDALL = "kendall"

    @property
    def calculator(self) -> Type[DistanceCalculator]:
        """
        Calculator class for this metric.

        Returns:
        --------
        Type[DistanceCalculator]
            Cosine, Correlation, Spearman, or Kendall distance calculator class.

        Raises:
        -------
        ValueError
            If the member is not mapped to a calculator.
        """
        match self:
            case SimilarityCorrelationDistanceMetric.COSINE:
                from .calculators.cosine import CosineDistanceCalculator

                return CosineDistanceCalculator
            case SimilarityCorrelationDistanceMetric.CORRELATION:
                from .calculators.correlation import CorrelationDistanceCalculator

                return CorrelationDistanceCalculator
            case SimilarityCorrelationDistanceMetric.SPEARMAN:
                from .calculators.spearman import SpearmanDistanceCalculator

                return SpearmanDistanceCalculator
            case SimilarityCorrelationDistanceMetric.KENDALL:
                from .calculators.kendall import KendallDistanceCalculator

                return KendallDistanceCalculator
        raise ValueError(f"No calculator found for metric {self}")
