"""
Minkowski-family metric names (Lp norms, Chebyshev, squared Euclidean).

Maps each enum member to a calculator class under calculators/.
"""

from enum import Enum
from typing import Type

from ...calculator import DistanceCalculator
from ...metric import DistanceMetric


class MinkowskiDistanceMetric(DistanceMetric, Enum):
    """
    Standard vector norms and related distances on equal-shaped samples.

    Members:
    --------
    MANHATTAN, EUCLIDEAN, MINKOWSKI, CHEBYSHEV, SQUARED_EUCLIDEAN
        String codes used by DistanceMetric.

    Notes:
    -----
    MINKOWSKI uses a user-configurable exponent on the calculator instance.
    """

    MANHATTAN = "manhattan"
    EUCLIDEAN = "euclidean"
    MINKOWSKI = "minkowski"
    CHEBYSHEV = "chebyshev"
    SQUARED_EUCLIDEAN = "squared_euclidean"

    @property
    def calculator(self) -> Type[DistanceCalculator]:
        """
        Calculator class for this metric.

        Returns:
        --------
        Type[DistanceCalculator]
            One of Manhattan, Euclidean, Minkowski, Chebyshev, or
            SquaredEuclidean distance calculator classes.

        Raises:
        -------
        ValueError
            If the member is not mapped to a calculator.
        """
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
