from enum import Enum
from typing import Type

from ...calculator import DistanceCalculator
from ...metric import DistanceMetric


class CovarianceNormalizedDistanceMetric(DistanceMetric, Enum):
    MAHALANOBIS = "mahalanobis"
    SEUCLIDEAN = "seuclidean"

    @property
    def calculator(self) -> Type[DistanceCalculator]:
        match self:
            case CovarianceNormalizedDistanceMetric.MAHALANOBIS:
                from .calculators.mahalanobis import MahalanobisDistanceCalculator

                return MahalanobisDistanceCalculator
            case CovarianceNormalizedDistanceMetric.SEUCLIDEAN:
                from .calculators.seuclidean import SEuclideanDistanceCalculator

                return SEuclideanDistanceCalculator
        raise ValueError(f"No calculator found for metric {self}")
