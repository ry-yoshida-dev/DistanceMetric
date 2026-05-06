"""
Kullback–Leibler divergence with epsilon smoothing.

Query is treated as the first distribution mass and gallery as the second.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...cross_elementwise import CrossElementwiseCalculatorBase

if TYPE_CHECKING:
    from ..metric import InformationTheoreticDistanceMetric


class KLDivergenceDistanceCalculator(CrossElementwiseCalculatorBase):
    """
    Directed KL divergence between nonnegative mass vectors.

    Notes:
    -----
    Asymmetric: swapping query and gallery changes the result. Pass eps through cross kwargs.
    """

    def _elementwise_values(
        self,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        *,
        eps: float = 1e-12,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Per-bin KL contributions before summing.

        Parameters:
        ----------
        query_array: np.ndarray
            Mass p.
        gallery_array: np.ndarray
            Mass q.
        eps: float, optional
            Floor applied to both p and q before ratios and logs.
        **kwargs: Any
            Unused.

        Returns:
        --------
        np.ndarray
            Element-wise KL contributions.
        """
        self._validate_broadcast_compatible(
            query_array=query_array,
            gallery_array=gallery_array,
        )
        eps = float(eps)
        p = np.maximum(query_array, eps)
        q = np.maximum(gallery_array, eps)
        return p * np.log(p / q)

    def _reduce_elementwise_values(
        self,
        values: np.ndarray,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Sum KL contributions across bins.

        Parameters:
        ----------
        values: np.ndarray
            Per-bin terms in cross layout.
        query_array: np.ndarray
            Unused.
        gallery_array: np.ndarray
            Unused.
        **kwargs: Any
            Unused.

        Returns:
        --------
        np.ndarray
            Shape (n, m) KL divergences.
        """
        return np.sum(values, axis=self._sample_value_axes(values))

    @property
    def metric(self) -> InformationTheoreticDistanceMetric:
        """
        Enum tag for this calculator.

        Returns:
        --------
        InformationTheoreticDistanceMetric
            Always KL_DIVERGENCE.
        """
        from ..metric import InformationTheoreticDistanceMetric
        return InformationTheoreticDistanceMetric.KL_DIVERGENCE
