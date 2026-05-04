from __future__ import annotations

from typing import TYPE_CHECKING

from ..base import MinkowskiDistanceCalculatorBase

if TYPE_CHECKING:
    from ..metric import MinkowskiDistanceMetric


class ManhattanDistanceCalculator(MinkowskiDistanceCalculatorBase):
    """Manhattan distance calculator (Minkowski with fixed p=1)."""

    @property
    def norm_order(self) -> float:
        return 1.0

    @property
    def metric(self) -> MinkowskiDistanceMetric:
        from ..metric import MinkowskiDistanceMetric

        return MinkowskiDistanceMetric.MANHATTAN
