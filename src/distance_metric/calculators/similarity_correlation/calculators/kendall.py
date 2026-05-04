from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

if TYPE_CHECKING:
    from ..metric import SimilarityCorrelationDistanceMetric


class KendallDistanceCalculator(CrossElementwiseCalculatorBase):
    def elementwise(
        self, query_array: np.ndarray, gallery_array: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        self._validate_same_shape(query_array, gallery_array)
        n = query_array.shape[-1]
        i_idx, j_idx = np.triu_indices(n, k=1)
        dq = query_array[..., i_idx] - query_array[..., j_idx]
        dg = gallery_array[..., i_idx] - gallery_array[..., j_idx]
        return np.sign(dq * dg).astype(float)

    def _reduce_elementwise_values(
        self,
        values: np.ndarray,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        concordant = np.sum(values > 0, axis=self._sample_value_axes(values)).astype(float)
        discordant = np.sum(values < 0, axis=self._sample_value_axes(values)).astype(float)
        denom = concordant + discordant
        tau = (concordant - discordant) / np.maximum(denom, 1e-12)
        return 1.0 - tau

    @property
    def metric(self) -> SimilarityCorrelationDistanceMetric:
        from ..metric import SimilarityCorrelationDistanceMetric

        return SimilarityCorrelationDistanceMetric.KENDALL
