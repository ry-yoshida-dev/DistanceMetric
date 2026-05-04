from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

if TYPE_CHECKING:
    from ..metric import CovarianceNormalizedDistanceMetric


class SEuclideanDistanceCalculator(CrossElementwiseCalculatorBase):
    def elementwise(
        self,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        *,
        variance: np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        self._validate_same_shape(query_array, gallery_array)
        if variance is None:
            raise ValueError("SEuclidean distance requires variance.")
        variance = np.asarray(variance, dtype=float)
        expected_shape = query_array.shape[2:] if query_array.ndim >= 3 else query_array.shape
        if variance.shape != expected_shape:
            raise ValueError("variance must have the same shape as each sample.")
        d = query_array - gallery_array
        return (d * d) / np.maximum(variance, 1e-12)

    def _reduce_elementwise_values(
        self,
        values: np.ndarray,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        *,
        variance: np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        return np.sqrt(np.sum(values, axis=self._sample_value_axes(values)))

    @property
    def metric(self) -> CovarianceNormalizedDistanceMetric:
        from ..metric import CovarianceNormalizedDistanceMetric

        return CovarianceNormalizedDistanceMetric.SEUCLIDEAN
