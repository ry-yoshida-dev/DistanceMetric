"""
Bases for cosine distance and correlation-based distances.

Cosine uses dot products and vector norms; correlation centers vectors; Spearman ranks within each row first.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..cross_elementwise import CrossElementwiseCalculatorBase


class CosineDistanceCalculatorBase(CrossElementwiseCalculatorBase):
    """
    Cosine distance 1 minus cosine similarity along the last axis.

    Notes:
    -----
    _cross_array is overridden to insert query and gallery batch axes explicitly.
    """

    def _elementwise_values(
        self,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Element-wise products before summing into a dot product.

        Parameters:
        ----------
        query_array: np.ndarray
            Query tensor broadcast-compatible with gallery_array.
        gallery_array: np.ndarray
            Gallery tensor matching query_array rules.
        **kwargs: Any
            Unused.

        Returns:
        --------
        np.ndarray
            query_array * gallery_array with broadcast shape.
        """
        self._validate_same_shape(query_array, gallery_array)
        return query_array * gallery_array

    def _reduce_elementwise_values(
        self,
        values: np.ndarray,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Turn dot products and L2 norms into cosine distance per batch pair.

        Parameters:
        ----------
        values: np.ndarray
            Element-wise products in cross layout (n, m, *features).
        query_array: np.ndarray
            Original query rows (n, *features) for norms.
        gallery_array: np.ndarray
            Original gallery rows (m, *features) for norms.
        **kwargs: Any
            Unused.

        Returns:
        --------
        np.ndarray
            Shape (n, m) cosine distances.
        """
        dot = np.sum(values, axis=self._sample_value_axes(values))
        q_norm = np.linalg.norm(query_array, axis=-1)
        g_norm = np.linalg.norm(gallery_array, axis=-1)
        denom = q_norm[:, None] * g_norm[None, :]
        return 1.0 - (dot / np.maximum(denom, 1e-12))

    def _cross_array(
        self, query_array: np.ndarray, gallery_array: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        """
        Broadcast cross layout then reduce to cosine distances.

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
            Cosine distance matrix (n, m) as float.
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


class CorrelationDistanceCalculatorBase(CosineDistanceCalculatorBase):
    """
    Pearson correlation expressed as distance after centering.

    Notes:
    -----
    _elementwise_values subtracts global means over the broadcast slice for generic paths.
    _cross_array recenters per row on the last axis before delegating to the cosine path.
    """

    def _elementwise_values(
        self, query_array: np.ndarray, gallery_array: np.ndarray, **kwargs: Any
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
        self._validate_same_shape(query_array, gallery_array)
        qc = query_array - np.mean(query_array)
        gc = gallery_array - np.mean(gallery_array)
        return qc * gc

    def _cross_array(
        self, query_array: np.ndarray, gallery_array: np.ndarray, **kwargs: Any
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


class SpearmanDistanceCalculatorBase(CorrelationDistanceCalculatorBase):
    """
    Spearman correlation distance via rank transforms along each row.

    Notes:
    -----
    Ranks use argsort twice (average ranks for ties are not implemented).
    """

    @staticmethod
    def _rank_values(values: np.ndarray) -> np.ndarray:
        """
        Ordinal ranks for a one-dimensional vector.

        Parameters:
        ----------
        values: np.ndarray
            One-dimensional sample along a row.

        Returns:
        --------
        np.ndarray
            Float ranks via argsort(argsort).
        """
        return np.argsort(np.argsort(values)).astype(float)

    def _cross_array(
        self, query_array: np.ndarray, gallery_array: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        """
        Rank each row, then correlation distance on ranks.

        Parameters:
        ----------
        query_array: np.ndarray
            Two-dimensional or higher; axis 1 is rank axis.
        gallery_array: np.ndarray
            Same row layout as query_array.
        **kwargs: Any
            Forwarded.

        Returns:
        --------
        np.ndarray
            Spearman-based distance matrix (n, m).
        """
        query_rank = np.apply_along_axis(self._rank_values, 1, query_array)
        gallery_rank = np.apply_along_axis(self._rank_values, 1, gallery_array)
        return super()._cross_array(query_rank, gallery_rank, **kwargs)
