from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np

from ..calculator import DistanceCalculator


class CrossElementwiseCalculatorBase(DistanceCalculator):
    """
    Generic calculator for metrics that can be expressed as:
    1) element-wise values for each (query, gallery) pair
    2) reduction of those values into a scalar distance.

    Subclasses only need to implement:
    - elementwise: returns per-dimension values.
    - _reduce_elementwise_values: reduces values over sample dimensions.
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
            Reduced distance values. 
            Typically shape (n, m), but scalar outputs are also supported for specialized metrics.
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
        # values is expected to be (n, m, *sample_dims) in cross mode.
        return tuple(range(2, values.ndim))

    def cross(
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
            Metric-specific parameters forwarded to elementwise and
            _reduce_elementwise_values.
        """
        np.broadcast_shapes(query_array.shape[1:], gallery_array.shape[1:])
        values = self.elementwise(query_array[:, None, ...], gallery_array[None, ...], **kwargs)
        return np.asarray(
            self._reduce_elementwise_values(
                values=values,
                query_array=query_array,
                gallery_array=gallery_array,
                **kwargs,
            ),
            dtype=float,
        )
