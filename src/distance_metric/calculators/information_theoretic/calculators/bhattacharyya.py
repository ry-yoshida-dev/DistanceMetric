"""
Bhattacharyya distance from nonnegative mass vectors.

Negative entries are floored to zero before geometric-mean and log steps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

if TYPE_CHECKING:
    from ..metric import InformationTheoreticDistanceMetric


class BhattacharyyaDistanceCalculator(CrossElementwiseCalculatorBase):
    """
    Bhattacharyya distance from overlap of nonnegative masses per bin.

    Notes:
    -----
    Larger values indicate less overlap between the two mass vectors.
    """

    def _elementwise_values(
        self,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Per-bin overlap terms before summing and taking negative log.

        Parameters:
        ----------
        query_array: np.ndarray
            Nonnegative mass; treated as p.
        gallery_array: np.ndarray
            Nonnegative mass; treated as q.
        **kwargs: Any
            Unused.

        Returns:
        --------
        np.ndarray
            Element-wise nonnegative overlap terms (float).
        """
        self._validate_broadcast_compatible(
            query_array=query_array,
            gallery_array=gallery_array,
        )
        return np.sqrt(np.maximum(query_array, 0.0) * np.maximum(gallery_array, 0.0))

    def _reduce_elementwise_values(
        self,
        values: np.ndarray,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Sum geometric-mean terms and apply negative log.

        Parameters:
        ----------
        values: np.ndarray
            Per-bin overlap terms in cross layout.
        query_array: np.ndarray
            Unused.
        gallery_array: np.ndarray
            Unused.
        **kwargs: Any
            Unused.

        Returns:
        --------
        np.ndarray
            Shape (n, m) Bhattacharyya distance values.
        """
        coef = np.sum(values, axis=self._sample_value_axes(values))
        return -np.log(np.maximum(coef, 1e-12))

    @property
    def metric(self) -> InformationTheoreticDistanceMetric:
        """
        Enum tag for this calculator.

        Returns:
        --------
        InformationTheoreticDistanceMetric
            Always BHATTACHARYYA.
        """
        from ..metric import InformationTheoreticDistanceMetric

        return InformationTheoreticDistanceMetric.BHATTACHARYYA
