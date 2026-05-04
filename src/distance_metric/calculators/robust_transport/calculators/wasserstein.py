from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

if TYPE_CHECKING:
    from ..metric import RobustTransportDistanceMetric


class WassersteinDistanceCalculator(CrossElementwiseCalculatorBase):
    def elementwise(
        self, query_array: np.ndarray, gallery_array: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        self._validate_same_shape(query_array, gallery_array)
        q = np.sort(query_array, axis=-1)
        g = np.sort(gallery_array, axis=-1)
        return np.abs(q - g)

    def _reduce_elementwise_values(
        self,
        values: np.ndarray,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        return np.mean(values, axis=self._sample_value_axes(values))

    @property
    def metric(self) -> RobustTransportDistanceMetric:
        from ..metric import RobustTransportDistanceMetric

        return RobustTransportDistanceMetric.WASSERSTEIN
