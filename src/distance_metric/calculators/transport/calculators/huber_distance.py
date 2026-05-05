"""
Huber loss summed across coordinates.

Quadratic for small residuals and linear beyond threshold delta.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

if TYPE_CHECKING:
    from ..metric import TransportDistanceMetric


class HuberDistanceCalculator(CrossElementwiseCalculatorBase):
    """
    Sum of per-coordinate Huber losses with threshold delta.

    Notes:
    -----
    Pass delta through cross, pairwise, or elementwise kwargs.
    """

    def _elementwise_values(
        self,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        *,
        delta: float = 1.0,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Huber kernel values per coordinate.

        Parameters:
        ----------
        query_array: np.ndarray
            Query batch matching gallery_array.
        gallery_array: np.ndarray
            Gallery batch matching query_array.
        delta: float, optional
            Threshold between quadratic and linear regions; must be positive.
        **kwargs: Any
            Unused.

        Returns:
        --------
        np.ndarray
            Per-coordinate Huber loss contributions.

        Raises:
        -------
        ValueError
            If delta is not positive.
        """
        self._validate_same_shape(query_array, gallery_array)
        delta = float(delta)
        if delta <= 0:
            raise ValueError("delta must be greater than 0.")
        a = np.abs(query_array - gallery_array)
        return np.where(a <= delta, 0.5 * a * a, delta * (a - 0.5 * delta))

    def _reduce_elementwise_values(
        self,
        values: np.ndarray,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Sum Huber losses over feature axes.

        Parameters:
        ----------
        values: np.ndarray
            Per-coordinate Huber terms in cross layout.
        query_array: np.ndarray
            Unused.
        gallery_array: np.ndarray
            Unused.
        **kwargs: Any
            Unused.

        Returns:
        --------
        np.ndarray
            Shape (n, m) total Huber distances.
        """
        return np.sum(values, axis=self._sample_value_axes(values))

    @property
    def metric(self) -> TransportDistanceMetric:
        """
        Enum tag for this calculator.

        Returns:
        --------
        TransportDistanceMetric
            Always HUBER_DISTANCE.
        """
        from ..metric import TransportDistanceMetric

        return TransportDistanceMetric.HUBER_DISTANCE
