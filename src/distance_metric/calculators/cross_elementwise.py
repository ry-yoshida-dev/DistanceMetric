"""
Cross-elementwise distance computation pattern.

Description:
------------
Many distances decompose into per-coordinate terms that are then aggregated
(sum, max, mean, …) over the feature dimensions of each sample.
CrossElementwiseCalculatorBase implements _cross_array by broadcasting a query
batch with shape (n, *) against a gallery batch with shape (m, *), computing
terms for every pair, and reducing to an (n, m) matrix.

Notes:
-----
Subclasses override _elementwise_values and _reduce_elementwise_values. Public
pairwise, cross, and elementwise from DistanceCalculator wrap numeric output in
DistanceResult as usual.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np

from ..calculator import DistanceCalculator


class CrossElementwiseCalculatorBase(DistanceCalculator):
    """
    Base class for metrics built from broadcastable per-coordinate terms plus a
    reduction over sample (feature) axes.

    Workflow:
    ---------
    1. _elementwise_values computes terms for broadcast-compatible
       query_array and gallery_array (same rules as NumPy broadcasting).
    2. _reduce_elementwise_values folds those terms into one scalar per
       (query row, gallery row) pair, typically shape (n, m).

    Override _cross_array only when the default broadcast-and-reduce flow is
    insufficient (for example non-separable metrics or custom batching).
    """

    @abstractmethod
    def _elementwise_values(
        self,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Per-coordinate terms before aggregation.

        Parameters:
        ----------
        query_array: np.ndarray
            Query tensor broadcast-compatible with gallery_array for this call.
        gallery_array: np.ndarray
            Gallery tensor; shapes must follow NumPy broadcasting with query_array.
        **kwargs: Any
            Metric-specific options forwarded from pairwise, cross, or elementwise.

        Returns:
        --------
        np.ndarray
            Terms that _reduce_elementwise_values will aggregate. In cross mode the
            caller inserts batch axes so typical shapes are (n, m, *feature_dims).

        Notes:
        -----
        Implementations usually call _validate_same_shape when inputs must align.
        """

    @abstractmethod
    def _reduce_elementwise_values(
        self,
        values: np.ndarray,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any
    ) -> np.ndarray | float:
        """
        Reduce broadcasted element-wise values into final distance scores.

        Parameters:
        ----------
        values: np.ndarray
            Element-wise values computed on broadcasted inputs with shape
            (n, m, *sample_shape).
        query_array: np.ndarray
            Original query batch with shape (n, *sample_shape).
        gallery_array: np.ndarray
            Original gallery batch with shape (m, *sample_shape).
        **kwargs: Any
            Metric-specific parameters forwarded from cross.

        Returns:
        --------
        np.ndarray | float
            Reduced distance values. Usually shape (n, m); some metrics
            (e.g. DTW) may return a scalar container for a single pair.
        """

    @staticmethod
    def _sample_value_axes(values: np.ndarray) -> tuple[int, ...]:
        """
        Return axes corresponding to sample dimensions in cross mode values.

        Parameters:
        ----------
        values: np.ndarray
            Broadcasted element-wise values with shape (n, m, *sample_shape).

        Returns:
        --------
        tuple[int, ...]
            Axes to reduce over when aggregating per-sample values.
        """
        # Cross mode stacks batch axes first: (n, m, *feature_dims).
        return tuple(range(2, values.ndim))

    def _cross_array(
        self,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Compute cross distances with broadcasting.

        Parameters:
        ----------
        query_array: np.ndarray
            Query batch with shape (n, *sample_shape).
        gallery_array: np.ndarray
            Gallery batch with shape (m, *sample_shape).
        **kwargs: Any
            Metric-specific parameters forwarded to _elementwise_values and
            _reduce_elementwise_values.

        Returns:
        --------
        np.ndarray
            Cross distance matrix of shape (n, m) with dtype float, unless a
            subclass overrides this method.

        Notes:
        -----
        Checks that sample_shape (trailing dimensions after the row axis) matches
        between query and gallery via numpy.broadcast_shapes on shape[1:].
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
