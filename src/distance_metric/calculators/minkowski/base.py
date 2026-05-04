from __future__ import annotations

from abc import abstractmethod

import numpy as np

from ..cross_elementwise import CrossElementwiseCalculatorBase


class MinkowskiDistanceCalculatorBase(CrossElementwiseCalculatorBase):
    """Base implementation for Minkowski-family distances."""

    @property
    @abstractmethod
    def norm_order(self) -> float:
        """Minkowski norm order p (> 0)."""

    def elementwise(
        self, query_array: np.ndarray, gallery_array: np.ndarray, **kwargs: object
    ) -> np.ndarray:
        self._validate_same_shape(query_array, gallery_array)
        return np.abs(query_array - gallery_array) ** float(self.norm_order)

    def _reduce_elementwise_values(
        self,
        values: np.ndarray,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: object,
    ) -> np.ndarray:
        resolved_p = float(self.norm_order)
        return np.sum(values, axis=self._sample_value_axes(values)) ** (1.0 / resolved_p)
