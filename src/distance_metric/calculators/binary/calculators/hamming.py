"""
Hamming distance calculator for aligned vectors.

Counts mismatched coordinates between two samples and averages over feature axes.
Supports arbitrary numeric dtypes; inequality uses NumPy element-wise comparison.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

if TYPE_CHECKING:
    from ..metric import BinaryDistanceMetric


class HammingDistanceCalculator(CrossElementwiseCalculatorBase):
    """
    Normalized Hamming distance between equal-shaped samples.

    Each coordinate contributes 1 if query and gallery differ and 0 otherwise.
    The cross-elementwise base averages those indicators over all sample feature
    axes, yielding the fraction of disagreeing positions when the sample is a flat
    vector. Pairwise and cross entry points are inherited from DistanceCalculator.
    Batch cross layouts follow CrossElementwiseCalculatorBase (value shape (n, m)).
    """

    def _elementwise_values(
        self, query_array: np.ndarray, gallery_array: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        """
        Per-coordinate mismatch indicators before averaging.

        Parameters:
        ----------
        query_array: np.ndarray
            Query tensor; must broadcast with gallery_array.
        gallery_array: np.ndarray
            Gallery tensor with the same shape as query_array for this call.
        **kwargs: Any
            Reserved; unused.

        Returns:
        --------
        np.ndarray
            Float array of zeros and ones with the broadcast shape of the inputs,
            where 1 marks positions where query_array != gallery_array.
        """
        self._validate_same_shape(query_array, gallery_array)
        return (query_array != gallery_array).astype(float)

    def _reduce_elementwise_values(
        self,
        values: np.ndarray,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Mean mismatch over all feature axes for each query-gallery pair.

        Parameters:
        ----------
        values: np.ndarray
            Output of _elementwise_values. In cross mode, shape begins with
            (n, m, ...) where trailing axes are feature dimensions.
        query_array: np.ndarray
            Original query batch; not read in this implementation.
        gallery_array: np.ndarray
            Original gallery batch; not read in this implementation.
        **kwargs: Any
            Reserved; unused.

        Returns:
        --------
        np.ndarray
            Array of shape (n, m) containing the mean of values along every axis
            after the first two batch dimensions.
        """
        return np.mean(values, axis=self._sample_value_axes(values))

    @property
    def metric(self) -> BinaryDistanceMetric:
        """
        Enum member identifying this implementation.

        Returns:
        --------
        BinaryDistanceMetric
            Always BinaryDistanceMetric.HAMMING.
        """
        from ..metric import BinaryDistanceMetric

        return BinaryDistanceMetric.HAMMING
