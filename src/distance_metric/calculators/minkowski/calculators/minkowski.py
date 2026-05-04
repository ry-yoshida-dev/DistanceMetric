from __future__ import annotations

from typing import TYPE_CHECKING

from ..base import MinkowskiDistanceCalculatorBase

if TYPE_CHECKING:
    from ..metric import MinkowskiDistanceMetric


class MinkowskiDistanceCalculator(MinkowskiDistanceCalculatorBase):
    """
    Minkowski distance calculator with configurable norm order.

    By default, this class behaves as Euclidean distance (p=2).
    """

    def __init__(self, norm_order: float = 2.0) -> None:
        self._norm_order = norm_order

    @property
    def norm_order(self) -> float:
        return self._norm_order

    @property
    def metric(self) -> MinkowskiDistanceMetric:
        from ..metric import MinkowskiDistanceMetric

        return MinkowskiDistanceMetric.MINKOWSKI
