"""
Hellinger distance between nonnegative distributions on shared bins.

Clamps mass at zero before the square-root and squaring steps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

if TYPE_CHECKING:
    from ..metric import InformationTheoreticDistanceMetric


class HellingerDistanceCalculator(CrossElementwiseCalculatorBase):
    """
    Hellinger-style distance from nonnegative bin masses.

    Notes:
    -----
    Inputs are floored at zero before root and square operations.
    """

    def _elementwise_values(
        self,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Per-bin terms before summing and final aggregation.

        Parameters:
        ----------
        query_array: np.ndarray
            Nonnegative bin masses.
        gallery_array: np.ndarray
            Nonnegative bin masses aligned with query_array.
        **kwargs: Any
            Unused.

        Returns:
        --------
        np.ndarray
            Intermediate nonnegative terms aligned with broadcast bins.
        """
        self._validate_broadcast_compatible(
            query_array=query_array,
            gallery_array=gallery_array,
        )
        return (np.sqrt(np.maximum(query_array, 0.0)) - np.sqrt(np.maximum(gallery_array, 0.0))) ** 2

    def _reduce_elementwise_values(
        self,
        values: np.ndarray,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Aggregate per-bin terms into Hellinger distances for each batch pair.

        Parameters:
        ----------
        values: np.ndarray
            Per-bin intermediate terms in cross layout.
        query_array: np.ndarray
            Unused.
        gallery_array: np.ndarray
            Unused.
        **kwargs: Any
            Unused.

        Returns:
        --------
        np.ndarray
            Shape (n, m) Hellinger distances.
        """
        return np.sqrt(0.5 * np.sum(values, axis=self._sample_value_axes(values)))

    @property
    def metric(self) -> InformationTheoreticDistanceMetric:
        """
        Enum tag for this calculator.

        Returns:
        --------
        InformationTheoreticDistanceMetric
            Always HELLINGER.
        """
        from ..metric import InformationTheoreticDistanceMetric
        return InformationTheoreticDistanceMetric.HELLINGER
