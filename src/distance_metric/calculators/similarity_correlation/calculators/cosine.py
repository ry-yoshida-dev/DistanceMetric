"""
Cosine distance calculator.

Uses CosineDistanceCalculatorBase for all numerical work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..bases import CosineDistanceCalculatorBase

if TYPE_CHECKING:
    from ..metric import SimilarityCorrelationDistanceMetric


class CosineDistanceCalculator(CosineDistanceCalculatorBase):
    """
    Thin wrapper tying the implementation to the COSINE enum member.

    Notes:
    -----
    Inherits pairwise, cross, and elementwise from DistanceCalculator.
    """

    @property
    def metric(self) -> SimilarityCorrelationDistanceMetric:
        """
        Enum tag for this calculator.

        Returns:
        --------
        SimilarityCorrelationDistanceMetric
            Always COSINE.
        """
        from ..metric import SimilarityCorrelationDistanceMetric

        return SimilarityCorrelationDistanceMetric.COSINE
