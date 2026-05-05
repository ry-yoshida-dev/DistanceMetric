from __future__ import annotations

import numpy as np

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from .result import DistanceResult, DistanceResultType


class DistanceCalculator(ABC):
    """
    Base calculator. Numeric computation lives in _cross_array; public methods
    return DistanceResult with value and semantic type (distance vs similarity).
    """

    result_type: ClassVar[DistanceResultType] = DistanceResultType.DISTANCE

    def pairwise(self, array: np.ndarray, **kwargs: Any) -> DistanceResult:
        """
        Calculate the pairwise distance between all elements in the array.

        Parameters:
        ----------
        array: np.ndarray
            The array to calculate the pairwise distance between with shape (n, *).

        Returns:
        --------
        DistanceResult
            Pairwise matrix in the value field with shape (n, n).
        """
        return self.cross(array, array, **kwargs)

    def cross(
        self,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any,
    ) -> DistanceResult:
        """
        Calculate the cross distance between the query and gallery arrays.

        Parameters:
        ----------
        query_array: np.ndarray
            The query array to calculate the cross distance with shape (n, *).
        gallery_array: np.ndarray
            The gallery array to calculate the cross distance with shape (m, *).

        Returns:
        --------
        DistanceResult
            Cross matrix in the value field with shape (n, m).
        """
        return self._wrap(self._cross_array(query_array, gallery_array, **kwargs))

    @abstractmethod
    def _cross_array(
        self,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Return the raw numeric matrix (distance or similarity values).

        Subclasses implement computation here; the public cross method wraps
        the result in DistanceResult using the result_type class attribute.
        """

    def _wrap(self, value: np.ndarray | float) -> DistanceResult:
        return DistanceResult(value=np.asarray(value), type=self.result_type)

    def elementwise(
        self,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any,
    ) -> DistanceResult:
        """
        Compare two single samples and return their metric value.

        Parameters:
        ----------
        query_array: np.ndarray
            Query sample (typically one sample, not batched).
        gallery_array: np.ndarray
            Gallery sample. Shapes must follow NumPy broadcasting rules with
            query_array; they need not be identical.

        Returns:
        --------
        DistanceResult
            Scalar metric in the value field (typically 0-d array).
        """
        self._validate_same_shape(query_array, gallery_array)
        inner = self._cross_array(
            query_array=query_array[None, ...],
            gallery_array=gallery_array[None, ...],
            **kwargs,
        )
        inner_arr = np.asarray(inner)
        return self._wrap(inner_arr.reshape(inner_arr.shape[0], inner_arr.shape[1])[0, 0])

    def _validate_same_shape(
        self,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
    ) -> None:
        """
        Ensure query_array and gallery_array are broadcast-compatible.

        Uses numpy.broadcast_shapes; raises ValueError if the shapes cannot be
        broadcast together. Identical shapes are not required.
        """
        np.broadcast_shapes(query_array.shape, gallery_array.shape)
