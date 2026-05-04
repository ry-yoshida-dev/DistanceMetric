from __future__ import annotations

from typing import TYPE_CHECKING

from ..base import MinkowskiDistanceCalculatorBase

if TYPE_CHECKING:
    from ..metric import MinkowskiDistanceMetric


class EuclideanDistanceCalculator(MinkowskiDistanceCalculatorBase):
    """Euclidean distance calculator (Minkowski with fixed p=2)."""

    @property
    def norm_order(self) -> float:
        return 2.0

    @property
    def metric(self) -> MinkowskiDistanceMetric:
        from ..metric import MinkowskiDistanceMetric

        return MinkowskiDistanceMetric.EUCLIDEAN
