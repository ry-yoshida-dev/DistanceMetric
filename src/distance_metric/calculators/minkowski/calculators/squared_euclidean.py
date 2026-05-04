from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

if TYPE_CHECKING:
    from ..metric import MinkowskiDistanceMetric


class SquaredEuclideanDistanceCalculator(CrossElementwiseCalculatorBase):
    """Squared Euclidean distance calculator."""

    def elementwise(
        self, query_array: np.ndarray, gallery_array: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        self._validate_same_shape(query_array, gallery_array)
        d = query_array - gallery_array
        return d * d

    def _reduce_elementwise_values(
        self,
        values: np.ndarray,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        return np.sum(values, axis=self._sample_value_axes(values))

    @property
    def metric(self) -> MinkowskiDistanceMetric:
        from ..metric import MinkowskiDistanceMetric

        return MinkowskiDistanceMetric.SQUARED_EUCLIDEAN
