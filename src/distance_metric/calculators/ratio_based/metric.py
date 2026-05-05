"""
Ratio-based distance metric names (Canberra, Bray-Curtis).

Suited to nonnegative abundance or count-like feature vectors.
"""

from enum import Enum

from ...calculator import DistanceCalculator
from ...metric import DistanceMetric


class RatioBasedDistanceMetric(DistanceMetric, Enum):
    """
    Relative-difference metrics with coupled numerators and denominators.

    Members:
    --------
    CANBERRA : str
        Sum of normalized absolute gaps per coordinate.
    BRAY_CURTIS : str
        Gap sum scaled by combined mass of both vectors.

    Notes:
    -----
    BrayCurtisDistanceCalculator overrides _cross_array to couple rows for the denominator.
    """

    CANBERRA = "canberra"
    BRAY_CURTIS = "bray_curtis"

    @property
    def calculator(self) -> DistanceCalculator:
        """
        Calculator instance for this metric.

        Returns:
        --------
        DistanceCalculator
            CanberraDistanceCalculator or BrayCurtisDistanceCalculator.

        Raises:
        -------
        ValueError
            If the member is not mapped to a calculator.
        """
        match self:
            case RatioBasedDistanceMetric.CANBERRA:
                from .calculators.canberra import CanberraDistanceCalculator

                return CanberraDistanceCalculator()
            case RatioBasedDistanceMetric.BRAY_CURTIS:
                from .calculators.bray_curtis import BrayCurtisDistanceCalculator

                return BrayCurtisDistanceCalculator()
        raise ValueError(f"No calculator found for metric {self}")
