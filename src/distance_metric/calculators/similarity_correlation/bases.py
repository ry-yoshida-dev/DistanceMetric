from __future__ import annotations

from typing import Any

import numpy as np

from ..cross_elementwise import CrossElementwiseCalculatorBase


class CosineDistanceCalculatorBase(CrossElementwiseCalculatorBase):
    def elementwise(
        self,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        self._validate_same_shape(query_array, gallery_array)
        return query_array * gallery_array

    def _reduce_elementwise_values(
        self,
        values: np.ndarray,
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        dot = np.sum(values, axis=self._sample_value_axes(values))
        q_norm = np.linalg.norm(query_array, axis=-1)
        g_norm = np.linalg.norm(gallery_array, axis=-1)
        denom = q_norm[:, None] * g_norm[None, :]
        return 1.0 - (dot / np.maximum(denom, 1e-12))

    def cross(
        self, query_array: np.ndarray, gallery_array: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
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


class CorrelationDistanceCalculatorBase(CosineDistanceCalculatorBase):
    def elementwise(
        self, query_array: np.ndarray, gallery_array: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        self._validate_same_shape(query_array, gallery_array)
        qc = query_array - np.mean(query_array)
        gc = gallery_array - np.mean(gallery_array)
        return qc * gc

    def cross(
        self, query_array: np.ndarray, gallery_array: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        np.broadcast_shapes(query_array.shape[1:], gallery_array.shape[1:])
        query_centered = query_array - np.mean(query_array, axis=-1, keepdims=True)
        gallery_centered = gallery_array - np.mean(gallery_array, axis=-1, keepdims=True)
        return super().cross(query_centered, gallery_centered, **kwargs)


class SpearmanDistanceCalculatorBase(CorrelationDistanceCalculatorBase):
    @staticmethod
    def _rank_values(values: np.ndarray) -> np.ndarray:
        return np.argsort(np.argsort(values)).astype(float)

    def cross(
        self, query_array: np.ndarray, gallery_array: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        query_rank = np.apply_along_axis(self._rank_values, 1, query_array)
        gallery_rank = np.apply_along_axis(self._rank_values, 1, gallery_array)
        return super().cross(query_rank, gallery_rank, **kwargs)
