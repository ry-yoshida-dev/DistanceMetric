from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

if TYPE_CHECKING:
    from ..metric import RatioBasedDistanceMetric


class BrayCurtisDistanceCalculator(CrossElementwiseCalculatorBase):
    def elementwise(
        self, query_array: np.ndarray, gallery_array: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        self._validate_same_shape(query_array, gallery_array)
        return np.abs(query_array - gallery_array)

    def _reduce_elementwise_values(
        self,
        values: np.ndarray,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        numerator = np.sum(values, axis=self._sample_value_axes(values))
        query_abs_sum = np.sum(np.abs(query_array), axis=tuple(range(1, query_array.ndim)))
        gallery_abs_sum = np.sum(np.abs(gallery_array), axis=tuple(range(1, gallery_array.ndim)))
        denominator = query_abs_sum[:, None] + gallery_abs_sum[None, :]
        return numerator / np.maximum(denominator, 1e-12)

    def cross(
        self, query_array: np.ndarray, gallery_array: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        np.broadcast_shapes(query_array.shape[1:], gallery_array.shape[1:])
        values = self.elementwise(query_array[:, None, ...], gallery_array[None, ...], **kwargs)
        return np.asarray(
            self._reduce_elementwise_values(
                values=values,
                query_array=query_array,
                gallery_array=gallery_array,
                **kwargs,
            ),
            dtype=float,
        )

    @property
    def metric(self) -> RatioBasedDistanceMetric:
        from ..metric import RatioBasedDistanceMetric

        return RatioBasedDistanceMetric.BRAY_CURTIS
