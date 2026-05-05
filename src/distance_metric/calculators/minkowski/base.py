"""
Minkowski (Lp) family distances built on the cross-elementwise pattern.

Subclasses supply the exponent via norm_order. Chebyshev distance uses a
separate calculator (max over coordinates, not this sum-and-root pattern).

Public pairwise, cross, and elementwise come from DistanceCalculator via
CrossElementwiseCalculatorBase; numeric work is in _elementwise_values and
_reduce_elementwise_values.
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from ..cross_elementwise import CrossElementwiseCalculatorBase


class MinkowskiDistanceCalculatorBase(CrossElementwiseCalculatorBase):
    """
    Base class for Lp distances using powered absolute differences.

    Mapping to hooks:
    -----------------
    _elementwise_values builds powered absolute differences on broadcast-compatible
    arrays. _reduce_elementwise_values sums along feature axes
    (CrossElementwiseCalculatorBase._sample_value_axes) and finishes the Minkowski
    reduction step for the batch matrix.

    Shape:
    ------
    In cross mode the parent inserts batch axes so inner arrays are shaped like
    (n, m, *features). Elementwise validation ensures query and gallery slices
    broadcast like identical feature tensors.

    Subclasses:
    -----------
    Provide norm_order only unless the metric definition must change.
    """

    @property
    @abstractmethod
    def norm_order(self) -> float:
        """
        Minkowski exponent (norm order).

        Returns:
        --------
        float
            Strictly positive value; used for powering differences and the final
            root step.

        Notes:
        -----
        Typical values are 1 (Manhattan) and 2 (Euclidean). The generic
        MinkowskiDistanceCalculator stores a user-supplied value here.
        """

    def _elementwise_values(
        self, query_array: np.ndarray, gallery_array: np.ndarray, **kwargs: object
    ) -> np.ndarray:
        """
        Powered absolute coordinate differences before reduction.

        Parameters:
        ----------
        query_array: np.ndarray
            Query tensor; must broadcast with gallery_array for this call.
        gallery_array: np.ndarray
            Gallery tensor with the same shape rules as query_array.
        **kwargs: object
            Unused for plain Lp; kept for compatibility with other calculators.

        Returns:
        --------
        np.ndarray
            Same shape as the broadcast of query and gallery; nonnegative float
            values suitable for summing then rooting.
        """
        self._validate_same_shape(query_array, gallery_array)
        return np.abs(query_array - gallery_array) ** float(self.norm_order)

    def _reduce_elementwise_values(
        self,
        values: np.ndarray,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: object,
    ) -> np.ndarray:
        """
        Aggregate powered differences over feature axes into batch distances.

        Parameters:
        ----------
        values: np.ndarray
            Output of _elementwise_values in cross layout (n, m, *feature_dims).
        query_array: np.ndarray
            Original query batch (n, *features); unused here.
        gallery_array: np.ndarray
            Original gallery batch (m, *features); unused here.
        **kwargs: object
            Unused for Lp.

        Returns:
        --------
        np.ndarray
            Matrix of shape (n, m) containing Minkowski distances.

        Notes:
        -----
        norm_order is read again so subclasses that allow changing p stay consistent.
        """
        resolved_p = float(self.norm_order)
        return np.sum(values, axis=self._sample_value_axes(values)) ** (1.0 / resolved_p)
