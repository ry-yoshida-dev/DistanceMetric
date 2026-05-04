from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

if TYPE_CHECKING:
    from ..metric import RatioBasedDistanceMetric


class CanberraDistanceCalculator(CrossElementwiseCalculatorBase):
    def elementwise(
        self, query_array: np.ndarray, gallery_array: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        self._validate_same_shape(query_array, gallery_array)
        num = np.abs(query_array - gallery_array)
        den = np.abs(query_array) + np.abs(gallery_array)
        return num / np.maximum(den, 1e-12)

    def _reduce_elementwise_values(
        self,
        values: np.ndarray,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        return np.sum(values, axis=self._sample_value_axes(values))

    @property
    def metric(self) -> RatioBasedDistanceMetric:
        from ..metric import RatioBasedDistanceMetric

        return RatioBasedDistanceMetric.CANBERRA
