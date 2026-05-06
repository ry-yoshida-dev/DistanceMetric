"""
Chebyshev (L-infinity) distance calculator.

Uses maximum absolute coordinate difference; separate from the Minkowski base class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

if TYPE_CHECKING:
    from ..metric import MinkowskiDistanceMetric


class ChebyshevDistanceCalculator(CrossElementwiseCalculatorBase):
    """
    Maximum absolute coordinate gap per sample pair (L-infinity).

    Notes:
    -----
    Implemented as max over absolute differences, not via the Minkowski sum hook.
    """

    def _elementwise_values(
        self,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Absolute coordinate-wise distances before taking the max.

        Parameters:
        ----------
        query_array: np.ndarray
            Query batch broadcast-compatible with gallery_array.
        gallery_array: np.ndarray
            Gallery batch with matching broadcast rules.
        **kwargs: Any
            Unused.

        Returns:
        --------
        np.ndarray
            Absolute differences with the same broadcast shape as inputs.
        """
        self._validate_broadcast_compatible(
            query_array=query_array,
            gallery_array=gallery_array,
        )
        return np.abs(query_array - gallery_array)

    def _reduce_elementwise_values(
        self,
        values: np.ndarray,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Maximum over all feature axes for each query-gallery pair.

        Parameters:
        ----------
        values: np.ndarray
            Absolute differences in cross layout starting with (n, m, ...).
        query_array: np.ndarray
            Unused.
        gallery_array: np.ndarray
            Unused.
        **kwargs: Any
            Unused.

        Returns:
        --------
        np.ndarray
            Shape (n, m); entry (i, j) is the Chebyshev distance between row i and row j.
        """
        return np.max(values, axis=self._sample_value_axes(values))

    @property
    def metric(self) -> MinkowskiDistanceMetric:
        """
        Enum tag for this calculator.

        Returns:
        --------
        MinkowskiDistanceMetric
            Always MinkowskiDistanceMetric.CHEBYSHEV.
        """
        from ..metric import MinkowskiDistanceMetric
        return MinkowskiDistanceMetric.CHEBYSHEV
