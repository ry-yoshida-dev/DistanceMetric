from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

if TYPE_CHECKING:
    from ..metric import InformationTheoreticDistanceMetric


class HellingerDistanceCalculator(CrossElementwiseCalculatorBase):
    def elementwise(
        self, query_array: np.ndarray, gallery_array: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        self._validate_same_shape(query_array, gallery_array)
        return (np.sqrt(np.maximum(query_array, 0.0)) - np.sqrt(np.maximum(gallery_array, 0.0))) ** 2

    def _reduce_elementwise_values(
        self,
        values: np.ndarray,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        return np.sqrt(0.5 * np.sum(values, axis=self._sample_value_axes(values)))

    @property
    def metric(self) -> InformationTheoreticDistanceMetric:
        from ..metric import InformationTheoreticDistanceMetric

        return InformationTheoreticDistanceMetric.HELLINGER
