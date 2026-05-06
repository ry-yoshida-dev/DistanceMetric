"""
Mahalanobis distance with inverse covariance or covariance matrix.

Combines each difference vector with vi (or pinv of cov) per batch pair.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

if TYPE_CHECKING:
    from ..metric import CovarianceNormalizedDistanceMetric


class MahalanobisDistanceCalculator(CrossElementwiseCalculatorBase):
    """
    Elliptical distance induced by a positive semidefinite Gram factor.

    Notes:
    -----
    Pass vi (inverse covariance) or cov (covariance, then pinv) through cross,
    pairwise, or elementwise. At least one of vi or cov is required in reduce.
    """

    def _elementwise_values(
        self,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        *,
        vi: np.ndarray | None = None,
        cov: np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Coordinate-wise differences before combining with vi.

        Parameters:
        ----------
        query_array: np.ndarray
            Query batch; must broadcast with gallery_array.
        gallery_array: np.ndarray
            Gallery batch; same shape rules as query_array.
        vi: np.ndarray, optional
            Passed through to reduce; not read here.
        cov: np.ndarray, optional
            Passed through to reduce; not read here.
        **kwargs: Any
            Additional options for subclasses; unused.

        Returns:
        --------
        np.ndarray
            query_array minus gallery_array, same broadcast shape.
        """
        self._validate_broadcast_compatible(
            query_array=query_array,
            gallery_array=gallery_array,
        )
        return query_array - gallery_array

    def _reduce_elementwise_values(
        self,
        values: np.ndarray,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        *,
        vi: np.ndarray | None = None,
        cov: np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Combine differences with vi and return nonnegative scalar distances per pair.

        Parameters:
        ----------
        values: np.ndarray
            Differences in cross layout (n, m, *d); last axis is feature index.
        query_array: np.ndarray
            Unused.
        gallery_array: np.ndarray
            Unused.
        vi: np.ndarray, optional
            Inverse covariance matrix, shape (d, d). Preferred if known.
        cov: np.ndarray, optional
            Covariance; vi is set to pinv(cov) when vi is None.
        **kwargs: Any
            Unused.

        Returns:
        --------
        np.ndarray
            Shape (n, m) non-negative Mahalanobis distances.

        Raises:
        -------
        ValueError
            If both vi and cov are None.
        """
        if vi is None:
            if cov is None:
                raise ValueError("Mahalanobis distance requires vi or cov.")
            vi = np.linalg.pinv(np.asarray(cov))
        vi = np.asarray(vi, dtype=float)
        quad = np.einsum("...i,ij,...j->...", values, vi, values)
        return np.sqrt(np.maximum(quad, 0.0))

    @property
    def metric(self) -> CovarianceNormalizedDistanceMetric:
        """
        Enum tag for this calculator.

        Returns:
        --------
        CovarianceNormalizedDistanceMetric
            Always MAHALANOBIS.
        """
        from ..metric import CovarianceNormalizedDistanceMetric

        return CovarianceNormalizedDistanceMetric.MAHALANOBIS
