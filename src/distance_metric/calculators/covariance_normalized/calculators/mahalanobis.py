from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

if TYPE_CHECKING:
    from ..metric import CovarianceNormalizedDistanceMetric


class MahalanobisDistanceCalculator(CrossElementwiseCalculatorBase):
    def elementwise(
        self,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        *,
        vi: np.ndarray | None = None,
        cov: np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        self._validate_same_shape(query_array, gallery_array)
        return query_array - gallery_array

    def _reduce_elementwise_values(
        self,
        values: np.ndarray,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        *,
        vi: np.ndarray | None = None,
        cov: np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        if vi is None:
            if cov is None:
                raise ValueError("Mahalanobis distance requires vi or cov.")
            vi = np.linalg.pinv(np.asarray(cov))
        vi = np.asarray(vi, dtype=float)
        quad = np.einsum("...i,ij,...j->...", values, vi, values)
        return np.sqrt(np.maximum(quad, 0.0))

    @property
    def metric(self) -> CovarianceNormalizedDistanceMetric:
        from ..metric import CovarianceNormalizedDistanceMetric

        return CovarianceNormalizedDistanceMetric.MAHALANOBIS
