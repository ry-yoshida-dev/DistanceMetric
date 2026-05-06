"""
Standardized Euclidean (SEuclidean) distance.

Divides squared differences by per-coordinate variance, then takes the square root
of the sum.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

if TYPE_CHECKING:
    from ..metric import CovarianceNormalizedDistanceMetric


class SEuclideanDistanceCalculator(CrossElementwiseCalculatorBase):
    """
    Euclidean distance with per-coordinate variance scaling.

    Notes:
    -----
    Supply variance on every call; its shape must match a single sample (trailing
    dimensions of the batch rows).
    """

    def _elementwise_values(
        self,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        *,
        variance: np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Per-coordinate scaled squared differences (d^2 / v).

        Parameters:
        ----------
        query_array: np.ndarray
            Query batch; must match gallery_array for broadcasting.
        gallery_array: np.ndarray
            Gallery batch; same shape rules.
        variance: np.ndarray, optional
            Per-coordinate variances. Required; must match each sample shape.
        **kwargs: Any
            Unused.

        Returns:
        --------
        np.ndarray
            (q - g)^2 / max(variance, tiny) with broadcast shape.

        Raises:
        -------
        ValueError
            If variance is None or its shape does not match each sample.
        """
        self._validate_broadcast_compatible(
            query_array=query_array,
            gallery_array=gallery_array,
        )
        if variance is None:
            raise ValueError("SEuclidean distance requires variance.")
        variance = np.asarray(variance, dtype=float)
        expected_shape = query_array.shape[2:] if query_array.ndim >= 3 else query_array.shape
        if variance.shape != expected_shape:
            raise ValueError("variance must have the same shape as each sample.")
        d = query_array - gallery_array
        return (d * d) / np.maximum(variance, 1e-12)

    def _reduce_elementwise_values(
        self,
        values: np.ndarray,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        *,
        variance: np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Square root of the sum of scaled squared differences.

        Parameters:
        ----------
        values: np.ndarray
            Scaled squares from _elementwise_values; cross shape (n, m, *features).
        query_array: np.ndarray
            Unused; variance was applied in the elementwise step.
        gallery_array: np.ndarray
            Unused.
        variance: np.ndarray, optional
            Echoed for signature compatibility; not read here.
        **kwargs: Any
            Unused.

        Returns:
        --------
        np.ndarray
            Shape (n, m) standardized Euclidean distances.
        """
        return np.sqrt(np.sum(values, axis=self._sample_value_axes(values)))

    @property
    def metric(self) -> CovarianceNormalizedDistanceMetric:
        """
        Enum tag for this calculator.

        Returns:
        --------
        CovarianceNormalizedDistanceMetric
            Always SEUCLIDEAN.
        """
        from ..metric import CovarianceNormalizedDistanceMetric

        return CovarianceNormalizedDistanceMetric.SEUCLIDEAN
