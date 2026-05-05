"""
Spearman rank-based correlation distance.

Ranks each row then applies CorrelationDistanceCalculatorBase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..bases import SpearmanDistanceCalculatorBase

if TYPE_CHECKING:
    from ..metric import SimilarityCorrelationDistanceMetric


class SpearmanDistanceCalculator(SpearmanDistanceCalculatorBase):
    """
    Enum-backed Spearman distance using rank-transformed rows.

    Notes:
    -----
    Inherits _cross_array from SpearmanDistanceCalculatorBase.
    """

    @property
    def metric(self) -> SimilarityCorrelationDistanceMetric:
        """
        Enum tag for this calculator.

        Returns:
        --------
        SimilarityCorrelationDistanceMetric
            Always SPEARMAN.
        """
        from ..metric import SimilarityCorrelationDistanceMetric

        return SimilarityCorrelationDistanceMetric.SPEARMAN
