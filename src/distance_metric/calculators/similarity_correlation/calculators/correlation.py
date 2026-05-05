"""
Pearson correlation distance calculator.

Uses CorrelationDistanceCalculatorBase for centering and normalization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..bases import CorrelationDistanceCalculatorBase

if TYPE_CHECKING:
    from ..metric import SimilarityCorrelationDistanceMetric


class CorrelationDistanceCalculator(CorrelationDistanceCalculatorBase):
    """
    Wrapper selecting SimilarityCorrelationDistanceMetric.CORRELATION.

    Notes:
    -----
    Cross-batch behavior follows CorrelationDistanceCalculatorBase._cross_array.
    """

    @property
    def metric(self) -> SimilarityCorrelationDistanceMetric:
        """
        Enum tag for this calculator.

        Returns:
        --------
        SimilarityCorrelationDistanceMetric
            Always CORRELATION.
        """
        from ..metric import SimilarityCorrelationDistanceMetric

        return SimilarityCorrelationDistanceMetric.CORRELATION
