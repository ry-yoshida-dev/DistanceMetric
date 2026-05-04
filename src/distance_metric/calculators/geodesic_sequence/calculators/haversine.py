from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

if TYPE_CHECKING:
    from ..metric import GeodesicSequenceDistanceMetric


class HaversineDistanceCalculator(CrossElementwiseCalculatorBase):
    def elementwise(
        self, query_array: np.ndarray, gallery_array: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        self._validate_same_shape(query_array, gallery_array)
        if query_array.shape[-1] != 2:
            raise ValueError("Haversine distance expects samples as [latitude, longitude].")
        lat1, lon1 = query_array[..., 0], query_array[..., 1]
        lat2, lon2 = gallery_array[..., 0], gallery_array[..., 1]
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        return np.asarray(a, dtype=float)[..., None]

    def _reduce_elementwise_values(
        self,
        values: np.ndarray,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        *,
        radius: float = 6371.0,
        **kwargs: Any,
    ) -> np.ndarray:
        radius = float(radius)
        a = values[..., 0]
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 0.0)))
        return radius * c

    @property
    def metric(self) -> GeodesicSequenceDistanceMetric:
        from ..metric import GeodesicSequenceDistanceMetric

        return GeodesicSequenceDistanceMetric.HAVERSINE
