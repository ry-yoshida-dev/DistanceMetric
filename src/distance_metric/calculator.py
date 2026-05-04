import numpy as np

from abc import ABC, abstractmethod
from typing import Any

class DistanceCalculator(ABC):
    def pairwise(self, array: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Calculate the pairwise distance between all elements in the array.

        Parameters:
        ----------
        array: np.ndarray
            The array to calculate the pairwise distance between with shape (n, *).

        Returns:
        --------
        np.ndarray
            The pairwise distance matrix with shape (n, n).
        """
        return self.cross(array, array, **kwargs)

    @abstractmethod
    def cross(
        self, 
        query_array: np.ndarray, 
        gallery_array: np.ndarray, 
        **kwargs: Any
    ) -> np.ndarray:
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
        np.ndarray
            The cross distance matrix with shape (n, m).
        """

    def elementwise(
        self, 
        query_array: np.ndarray, 
        gallery_array: np.ndarray, 
        **kwargs: Any
    ) -> np.ndarray:
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
        np.ndarray
            Metric value for the two samples.
        """
        self._validate_same_shape(query_array, gallery_array)
        return np.asarray(
            self.cross(
                query_array=query_array[None, ...],
                gallery_array=gallery_array[None, ...],
                **kwargs,
            )
        )[0, 0]

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

