"""
Bray–Curtis dissimilarity for nonnegative abundance-style vectors.

Uses absolute coordinate gaps in the numerator and row-wise mass in the denominator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

if TYPE_CHECKING:
    from ..metric import RatioBasedDistanceMetric


class BrayCurtisDistanceCalculator(CrossElementwiseCalculatorBase):
    """
    Bray–Curtis dissimilarity: absolute gap sum scaled by combined row mass.

    Notes:
    -----
    _cross_array is overridden so the denominator uses the correct query and gallery
    rows for each matrix entry.
    """

    def _elementwise_values(
        self,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Absolute coordinate-wise differences.

        Parameters:
        ----------
        query_array: np.ndarray
            Query batch matching gallery_array.
        gallery_array: np.ndarray
            Gallery batch matching query_array.
        **kwargs: Any
            Unused.

        Returns:
        --------
        np.ndarray
            |q - g| with broadcast shape.
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
        Ratio of summed absolute gaps to summed magnitudes per pair.

        Parameters:
        ----------
        values: np.ndarray
            Absolute differences in cross layout (n, m, *features).
        query_array: np.ndarray
            Original query rows (n, *features) for absolute mass sums per row.
        gallery_array: np.ndarray
            Original gallery rows (m, *features) for absolute mass sums per row.
        **kwargs: Any
            Unused.

        Returns:
        --------
        np.ndarray
            Shape (n, m) Bray–Curtis dissimilarities with epsilon in the denominator.
        """
        numerator = np.sum(values, axis=self._sample_value_axes(values))
        query_abs_sum = np.sum(np.abs(query_array), axis=tuple(range(1, query_array.ndim)))
        gallery_abs_sum = np.sum(np.abs(gallery_array), axis=tuple(range(1, gallery_array.ndim)))
        denominator = query_abs_sum[:, None] + gallery_abs_sum[None, :]
        return numerator / np.maximum(denominator, 1e-12)

    def _cross_array(
        self,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Same broadcast layout as CrossElementwiseCalculatorBase._cross_array.

        Parameters:
        ----------
        query_array: np.ndarray
            Shape (n, *sample_shape).
        gallery_array: np.ndarray
            Shape (m, *sample_shape).
        **kwargs: Any
            Forwarded to _elementwise_values and _reduce_elementwise_values.

        Returns:
        --------
        np.ndarray
            Bray–Curtis matrix of shape (n, m) as float.

        Notes:
        -----
        Explicit override delegates to reduce so denominator uses full query and gallery batches.
        """
        np.broadcast_shapes(query_array.shape[1:], gallery_array.shape[1:])
        values = self._elementwise_values(
            query_array[:, None, ...], gallery_array[None, ...], **kwargs
        )
        return np.asarray(
            self._reduce_elementwise_values(
                values=values,
                query_array=query_array,
                gallery_array=gallery_array,
                **kwargs,
            ),
            dtype=float,
        )

    @property
    def metric(self) -> RatioBasedDistanceMetric:
        """
        Enum tag for this calculator.

        Returns:
        --------
        RatioBasedDistanceMetric
            Always BRAY_CURTIS.
        """
        from ..metric import RatioBasedDistanceMetric

        return RatioBasedDistanceMetric.BRAY_CURTIS
