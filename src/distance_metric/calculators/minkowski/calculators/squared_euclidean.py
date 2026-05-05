"""
Squared Euclidean distance calculator.

Sums squared differences without taking the square root.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

if TYPE_CHECKING:
    from ..metric import MinkowskiDistanceMetric


class SquaredEuclideanDistanceCalculator(CrossElementwiseCalculatorBase):
    """
    Sum of squared coordinate differences (no square root).

    Notes:
    -----
    Often used where ranking by distance is enough without taking sqrt.
    """

    def _elementwise_values(
        self, query_array: np.ndarray, gallery_array: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        """
        Squared coordinate-wise gaps.

        Parameters:
        ----------
        query_array: np.ndarray
            Query batch broadcast-compatible with gallery_array.
        gallery_array: np.ndarray
            Gallery batch matching query_array broadcast rules.
        **kwargs: Any
            Unused.

        Returns:
        --------
        np.ndarray
            Element-wise (q - g)^2 with broadcast shape.
        """
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
        """
        Sum squared gaps over feature axes.

        Parameters:
        ----------
        values: np.ndarray
            Squared differences in cross layout (n, m, *features).
        query_array: np.ndarray
            Unused.
        gallery_array: np.ndarray
            Unused.
        **kwargs: Any
            Unused.

        Returns:
        --------
        np.ndarray
            Shape (n, m) containing summed squared Euclidean distances.
        """
        return np.sum(values, axis=self._sample_value_axes(values))

    @property
    def metric(self) -> MinkowskiDistanceMetric:
        """
        Enum tag for this calculator.

        Returns:
        --------
        MinkowskiDistanceMetric
            Always MinkowskiDistanceMetric.SQUARED_EUCLIDEAN.
        """
        from ..metric import MinkowskiDistanceMetric

        return MinkowskiDistanceMetric.SQUARED_EUCLIDEAN
