"""
Kendall-style distance from pairwise concordance of coordinate ordering.

Compares relative order of feature pairs between query and gallery samples.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

if TYPE_CHECKING:
    from ..metric import SimilarityCorrelationDistanceMetric


class KendallDistanceCalculator(CrossElementwiseCalculatorBase):
    """
    Concordance-based rank association converted to a distance.

    Notes:
    -----
    Uses upper-triangular index pairs along the last axis. Ties yield zero contribution.
    """

    def _elementwise_values(
        self, query_array: np.ndarray, gallery_array: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        """
        Concordance sign per unordered feature pair from upper-triangular indices.

        Parameters:
        ----------
        query_array: np.ndarray
            Samples with feature dimension last.
        gallery_array: np.ndarray
            Matching gallery samples.
        **kwargs: Any
            Unused.

        Returns:
        --------
        np.ndarray
            Values in {-1, 0, 1} indicating discordant, tie, or concordant pairs per bundle.
        """
        self._validate_same_shape(query_array, gallery_array)
        n = query_array.shape[-1]
        i_idx, j_idx = np.triu_indices(n, k=1)
        dq = query_array[..., i_idx] - query_array[..., j_idx]
        dg = gallery_array[..., i_idx] - gallery_array[..., j_idx]
        return np.sign(dq * dg).astype(float)

    def _reduce_elementwise_values(
        self,
        values: np.ndarray,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Map concordance and discordance counts to a distance in (0, 1) range.

        Parameters:
        ----------
        values: np.ndarray
            Pairwise sign indicators in cross layout.
        query_array: np.ndarray
            Unused.
        gallery_array: np.ndarray
            Unused.
        **kwargs: Any
            Unused.

        Returns:
        --------
        np.ndarray
            Shape (n, m) Kendall distances.

        Notes:
        -----
        Denominator uses concordant plus discordant counts with epsilon stability.
        """
        concordant = np.sum(values > 0, axis=self._sample_value_axes(values)).astype(float)
        discordant = np.sum(values < 0, axis=self._sample_value_axes(values)).astype(float)
        denom = concordant + discordant
        tau = (concordant - discordant) / np.maximum(denom, 1e-12)
        return 1.0 - tau

    @property
    def metric(self) -> SimilarityCorrelationDistanceMetric:
        """
        Enum tag for this calculator.

        Returns:
        --------
        SimilarityCorrelationDistanceMetric
            Always KENDALL.
        """
        from ..metric import SimilarityCorrelationDistanceMetric

        return SimilarityCorrelationDistanceMetric.KENDALL
