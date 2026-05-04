from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

if TYPE_CHECKING:
    from ..metric import SetMismatchDistanceMetric


class JaccardDistanceCalculator(CrossElementwiseCalculatorBase):
    def elementwise(
        self, query_array: np.ndarray, gallery_array: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        self._validate_same_shape(query_array, gallery_array)
        q = query_array.astype(bool)
        g = gallery_array.astype(bool)
        return np.stack([np.logical_and(q, g), np.logical_or(q, g)], axis=2).astype(float)

    def _reduce_elementwise_values(
        self,
        values: np.ndarray,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        sample_axes = tuple(range(3, values.ndim))
        inter = np.sum(values[:, :, 0, ...], axis=sample_axes)
        union = np.sum(values[:, :, 1, ...], axis=sample_axes)
        return 1.0 - (inter / np.maximum(union, 1e-12))

    @property
    def metric(self) -> SetMismatchDistanceMetric:
        from ..metric import SetMismatchDistanceMetric

        return SetMismatchDistanceMetric.JACCARD
