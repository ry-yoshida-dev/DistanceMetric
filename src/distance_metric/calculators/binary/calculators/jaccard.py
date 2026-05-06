"""
Jaccard distance calculator for binary feature vectors.

Nonzero entries are treated as set membership after boolean conversion. Batch
cross mode compares every query row to every gallery row via the cross-elementwise
machinery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

if TYPE_CHECKING:
    from ..metric import BinaryDistanceMetric


class JaccardDistanceCalculator(CrossElementwiseCalculatorBase):
    """
    Jaccard distance on bit- or boolean-like vectors.

    Intersection and union are accumulated per coordinate channel, then combined
    into a scalar per query-gallery pair. Values are coerced with astype(bool).

    In cross mode the parent broadcasts query rows (n, *) against gallery rows
    (m, *); this implementation stacks channels along axis 2 and sums over feature
    dimensions.
    """

    def _elementwise_values(
        self,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Build per-coordinate intersection and union indicators for reduction.

        Parameters:
        ----------
        query_array: np.ndarray
            Query batch, broadcast-compatible with gallery_array. Non-boolean
            values are interpreted as false (0) or true (nonzero) per element.
        gallery_array: np.ndarray
            Gallery batch with the same shape as query_array for this call.
        **kwargs: Any
            Reserved; unused.

        Returns:
        --------
        np.ndarray
            Stacked array with an axis at index 2: channel 0 is logical AND,
            channel 1 is logical OR (float-cast), per coordinate.
        """
        self._validate_broadcast_compatible(
            query_array=query_array,
            gallery_array=gallery_array,
        )
        q = query_array.astype(bool)
        g = gallery_array.astype(bool)
        return np.stack([np.logical_and(q, g), np.logical_or(q, g)], axis=2).astype(float)

    def _reduce_elementwise_values(
        self,
        values: np.ndarray,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Aggregate channels into Jaccard distance per batch pair.

        Parameters:
        ----------
        values: np.ndarray
            Output of _elementwise_values in cross layout: batch n, batch m,
            channel axis (AND/OR), then feature dimensions.
        query_array: np.ndarray
            Original query batch; not read in this implementation.
        gallery_array: np.ndarray
            Original gallery batch; not read in this implementation.
        **kwargs: Any
            Reserved; unused.

        Returns:
        --------
        np.ndarray
            Shape (n, m). Denominator uses a small floor to avoid division by zero.

        Notes:
        -----
        After selecting channel 0 or 1, remaining axes are query batch, gallery
        batch, then feature axes to sum out before dividing intersection by union.
        """
        intersection_plane = values[:, :, 0, ...]
        union_plane = values[:, :, 1, ...]
        sample_axes = tuple(range(2, intersection_plane.ndim))
        inter = np.sum(intersection_plane, axis=sample_axes)
        union = np.sum(union_plane, axis=sample_axes)
        return 1.0 - (inter / np.maximum(union, 1e-12))

    @property
    def metric(self) -> BinaryDistanceMetric:
        """
        Enum member identifying this implementation.

        Returns:
        --------
        BinaryDistanceMetric
            Always BinaryDistanceMetric.JACCARD.
        """
        from ..metric import BinaryDistanceMetric
        return BinaryDistanceMetric.JACCARD
