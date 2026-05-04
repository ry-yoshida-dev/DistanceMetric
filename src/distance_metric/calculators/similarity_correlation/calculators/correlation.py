from __future__ import annotations

from typing import TYPE_CHECKING

from ..bases import CorrelationDistanceCalculatorBase

if TYPE_CHECKING:
    from ..metric import SimilarityCorrelationDistanceMetric


class CorrelationDistanceCalculator(CorrelationDistanceCalculatorBase):
    @property
    def metric(self) -> SimilarityCorrelationDistanceMetric:
        from ..metric import SimilarityCorrelationDistanceMetric

        return SimilarityCorrelationDistanceMetric.CORRELATION
