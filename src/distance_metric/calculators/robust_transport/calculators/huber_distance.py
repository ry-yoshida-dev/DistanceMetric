from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

if TYPE_CHECKING:
    from ..metric import RobustTransportDistanceMetric


class HuberDistanceCalculator(CrossElementwiseCalculatorBase):
    def elementwise(
        self,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        *,
        delta: float = 1.0,
        **kwargs: Any,
    ) -> np.ndarray:
        self._validate_same_shape(query_array, gallery_array)
        delta = float(delta)
        if delta <= 0:
            raise ValueError("delta must be greater than 0.")
        a = np.abs(query_array - gallery_array)
        return np.where(a <= delta, 0.5 * a * a, delta * (a - 0.5 * delta))

    def _reduce_elementwise_values(
        self,
        values: np.ndarray,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        return np.sum(values, axis=self._sample_value_axes(values))

    @property
    def metric(self) -> RobustTransportDistanceMetric:
        from ..metric import RobustTransportDistanceMetric

        return RobustTransportDistanceMetric.HUBER_DISTANCE
