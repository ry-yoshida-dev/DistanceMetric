from enum import Enum
from typing import Type

from ...calculator import DistanceCalculator
from ...metric import DistanceMetric


class MinkowskiDistanceMetric(DistanceMetric, Enum):
    MANHATTAN = "manhattan"
    EUCLIDEAN = "euclidean"
    MINKOWSKI = "minkowski"
    CHEBYSHEV = "chebyshev"
    SQUARED_EUCLIDEAN = "squared_euclidean"

    @property
    def calculator(self) -> Type[DistanceCalculator]:
        match self:
            case MinkowskiDistanceMetric.MANHATTAN:
                from .calculators.manhattan import ManhattanDistanceCalculator

                return ManhattanDistanceCalculator
            case MinkowskiDistanceMetric.EUCLIDEAN:
                from .calculators.euclidean import EuclideanDistanceCalculator

                return EuclideanDistanceCalculator
            case MinkowskiDistanceMetric.MINKOWSKI:
                from .calculators.minkowski import MinkowskiDistanceCalculator

                return MinkowskiDistanceCalculator
            case MinkowskiDistanceMetric.CHEBYSHEV:
                from .calculators.chebyshev import ChebyshevDistanceCalculator

                return ChebyshevDistanceCalculator
            case MinkowskiDistanceMetric.SQUARED_EUCLIDEAN:
                from .calculators.squared_euclidean import SquaredEuclideanDistanceCalculator

                return SquaredEuclideanDistanceCalculator
        raise ValueError(f"No calculator found for metric {self}")

