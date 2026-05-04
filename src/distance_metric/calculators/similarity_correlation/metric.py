from enum import Enum
from typing import Type

from ...calculator import DistanceCalculator
from ...metric import DistanceMetric


class SimilarityCorrelationDistanceMetric(DistanceMetric, Enum):
    COSINE = "cosine"
    CORRELATION = "correlation"
    SPEARMAN = "spearman"
    KENDALL = "kendall"

    @property
    def calculator(self) -> Type[DistanceCalculator]:
        match self:
            case SimilarityCorrelationDistanceMetric.COSINE:
                from .calculators.cosine import CosineDistanceCalculator

                return CosineDistanceCalculator
            case SimilarityCorrelationDistanceMetric.CORRELATION:
                from .calculators.correlation import CorrelationDistanceCalculator

                return CorrelationDistanceCalculator
            case SimilarityCorrelationDistanceMetric.SPEARMAN:
                from .calculators.spearman import SpearmanDistanceCalculator

                return SpearmanDistanceCalculator
            case SimilarityCorrelationDistanceMetric.KENDALL:
                from .calculators.kendall import KendallDistanceCalculator

                return KendallDistanceCalculator
        raise ValueError(f"No calculator found for metric {self}")
