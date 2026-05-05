"""
Covariance-normalized distance metric names (Mahalanobis, standardized Euclidean).

Calculators require extra arrays (inverse covariance or variances) at call time.
"""

from enum import Enum
from typing import Type

from ...calculator import DistanceCalculator
from ...metric import DistanceMetric


class CovarianceNormalizedDistanceMetric(DistanceMetric, Enum):
    """
    Metrics that weight coordinates by estimated scale or full covariance.

    Members:
    --------
    MAHALANOBIS : str
        Quadratic form with inverse covariance or pseudoinverse of cov.
    SEUCLIDEAN : str
        Per-feature variance scaling inside Euclidean length.

    Notes:
    -----
    Pass vi or cov, or variance, through pairwise, cross, or elementwise kwargs.
    """

    MAHALANOBIS = "mahalanobis"
    SEUCLIDEAN = "seuclidean"

    @property
    def calculator(self) -> Type[DistanceCalculator]:
        """
        Calculator class for this metric.

        Returns:
        --------
        Type[DistanceCalculator]
            MahalanobisDistanceCalculator or SEuclideanDistanceCalculator.

        Raises:
        -------
        ValueError
            If the member is not mapped to a calculator.
        """
        match self:
            case CovarianceNormalizedDistanceMetric.MAHALANOBIS:
                from .calculators.mahalanobis import MahalanobisDistanceCalculator

                return MahalanobisDistanceCalculator
            case CovarianceNormalizedDistanceMetric.SEUCLIDEAN:
                from .calculators.seuclidean import SEuclideanDistanceCalculator

                return SEuclideanDistanceCalculator
        raise ValueError(f"No calculator found for metric {self}")
