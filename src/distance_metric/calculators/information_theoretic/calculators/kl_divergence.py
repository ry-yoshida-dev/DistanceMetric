from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

if TYPE_CHECKING:
    from ..metric import InformationTheoreticDistanceMetric


class KLDivergenceDistanceCalculator(CrossElementwiseCalculatorBase):
    def elementwise(
        self,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        *,
        eps: float = 1e-12,
        **kwargs: Any,
    ) -> np.ndarray:
        self._validate_same_shape(query_array, gallery_array)
        eps = float(eps)
        p = np.maximum(query_array, eps)
        q = np.maximum(gallery_array, eps)
        return p * np.log(p / q)

    def _reduce_elementwise_values(
        self,
        values: np.ndarray,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        return np.sum(values, axis=self._sample_value_axes(values))

    @property
    def metric(self) -> InformationTheoreticDistanceMetric:
        from ..metric import InformationTheoreticDistanceMetric

        return InformationTheoreticDistanceMetric.KL_DIVERGENCE
