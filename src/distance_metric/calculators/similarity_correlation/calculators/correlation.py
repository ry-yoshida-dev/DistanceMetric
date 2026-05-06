"""
Pearson correlation distance calculator.

Correlation distance centers each row on the last axis, then applies cosine distance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .cosine import CosineDistanceCalculator

if TYPE_CHECKING:
    from ..metric import SimilarityCorrelationDistanceMetric


class CorrelationDistanceCalculator(CosineDistanceCalculator):
    """
    Pearson correlation expressed as distance after centering.

    Notes:
    -----
    _elementwise_values subtracts global means over the broadcast slice for generic paths.
    _cross_array recenters per row on the last axis before delegating to the cosine path.
    """

    def _elementwise_values(
        self,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Centered product terms (q - mean(q)) * (g - mean(g)) with global means.

        Parameters:
        ----------
        query_array: np.ndarray
            Query tensor matching gallery_array.
        gallery_array: np.ndarray
            Gallery tensor matching query_array.
        **kwargs: Any
            Unused.

        Returns:
        --------
        np.ndarray
            Centered products on the broadcast shape.
        """
        self._validate_broadcast_compatible(
            query_array=query_array,
            gallery_array=gallery_array,
        )
        qc = query_array - np.mean(query_array)
        gc = gallery_array - np.mean(gallery_array)
        return qc * gc

    def _cross_array(
        self,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Per-row centering on the last axis, then cosine-distance machinery.

        Parameters:
        ----------
        query_array: np.ndarray
            Batch (n, *features).
        gallery_array: np.ndarray
            Batch (m, *features).
        **kwargs: Any
            Forwarded.

        Returns:
        --------
        np.ndarray
            Correlation distance matrix (n, m).
        """
        np.broadcast_shapes(query_array.shape[1:], gallery_array.shape[1:])
        query_centered = query_array - np.mean(query_array, axis=-1, keepdims=True)
        gallery_centered = gallery_array - np.mean(gallery_array, axis=-1, keepdims=True)
        return super()._cross_array(query_centered, gallery_centered, **kwargs)

    @property
    def metric(self) -> SimilarityCorrelationDistanceMetric:
        """
        Enum tag for this calculator.

        Returns:
        --------
        SimilarityCorrelationDistanceMetric
            Always CORRELATION.
        """
        from ..metric import SimilarityCorrelationDistanceMetric

        return SimilarityCorrelationDistanceMetric.CORRELATION
