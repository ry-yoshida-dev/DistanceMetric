from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

if TYPE_CHECKING:
    from ..metric import GeodesicSequenceDistanceMetric


class DynamicTimeWarpingDistanceCalculator(CrossElementwiseCalculatorBase):
    def elementwise(
        self, query_array: np.ndarray, gallery_array: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        q = np.asarray(query_array)
        g = np.asarray(gallery_array)
        n, m = q.shape[0], g.shape[0]
        if q.ndim == 1 and g.ndim == 1:
            cost = np.abs(q[:, None] - g[None, :])
        else:
            cost = np.linalg.norm(q[:, None, ...] - g[None, ...], axis=-1)
        dp = np.full((n + 1, m + 1), np.inf, dtype=float)
        dp[0, 0] = 0.0
        for k in range(2, n + m + 1):
            i = np.arange(max(1, k - m), min(n, k - 1) + 1)
            j = k - i
            dp[i, j] = cost[i - 1, j - 1] + np.minimum(
                np.minimum(dp[i - 1, j], dp[i, j - 1]),
                dp[i - 1, j - 1],
            )
        return np.asarray([dp[n, m]], dtype=float)

    def _reduce_elementwise_values(
        self,
        values: np.ndarray,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any,
    ) -> float:
        return float(values[0])

    def cross(
        self, query_array: np.ndarray, gallery_array: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        self._validate_same_shape(query_array, gallery_array)
        n, m = query_array.shape[0], gallery_array.shape[0]
        out = np.empty((n, m), dtype=float)
        for flat_idx in range(n * m):
            i, j = divmod(flat_idx, m)
            out[i, j] = self._reduce_elementwise_values(
                values=self.elementwise(query_array[i], gallery_array[j], **kwargs),
                query_array=query_array[i],
                gallery_array=gallery_array[j],
                **kwargs,
            )
        return out

    @property
    def metric(self) -> GeodesicSequenceDistanceMetric:
        from ..metric import GeodesicSequenceDistanceMetric

        return GeodesicSequenceDistanceMetric.DYNAMIC_TIME_WARPING
