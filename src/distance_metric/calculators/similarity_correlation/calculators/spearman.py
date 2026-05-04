from __future__ import annotations

from typing import TYPE_CHECKING

from ..bases import SpearmanDistanceCalculatorBase

if TYPE_CHECKING:
    from ..metric import SimilarityCorrelationDistanceMetric


class SpearmanDistanceCalculator(SpearmanDistanceCalculatorBase):
    @property
    def metric(self) -> SimilarityCorrelationDistanceMetric:
        from ..metric import SimilarityCorrelationDistanceMetric

        return SimilarityCorrelationDistanceMetric.SPEARMAN
