from __future__ import annotations

from typing import TYPE_CHECKING

from ..bases import CosineDistanceCalculatorBase

if TYPE_CHECKING:
    from ..metric import SimilarityCorrelationDistanceMetric


class CosineDistanceCalculator(CosineDistanceCalculatorBase):
    @property
    def metric(self) -> SimilarityCorrelationDistanceMetric:
        from ..metric import SimilarityCorrelationDistanceMetric

        return SimilarityCorrelationDistanceMetric.COSINE
