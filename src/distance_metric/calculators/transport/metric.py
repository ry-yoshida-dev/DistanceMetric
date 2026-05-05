"""
Transport-style metric names (Huber loss sum, 1-D Wasserstein).

Huber uses a delta threshold; Wasserstein sorts marginals along the last axis.
"""

from enum import Enum

from ...calculator import DistanceCalculator
from ...metric import DistanceMetric


class TransportDistanceMetric(DistanceMetric, Enum):
    """
    Huber loss and empirical transport distances.

    Members:
    --------
    HUBER_DISTANCE : str
        Sum of per-coordinate Huber losses between query and gallery samples.
    WASSERSTEIN : str
        Mean absolute gap between sorted coordinates (equal bin weights).

    Notes:
    -----
    Pass delta for Huber through cross kwargs.
    """

    HUBER_DISTANCE = "huber_distance"
    WASSERSTEIN = "wasserstein"

    @property
    def calculator(self) -> DistanceCalculator:
        """
        Calculator instance for this metric.

        Returns:
        --------
        DistanceCalculator
            HuberDistanceCalculator or WassersteinDistanceCalculator.

        Raises:
        -------
        ValueError
            If the member is not mapped to a calculator.
        """
        match self:
            case TransportDistanceMetric.HUBER_DISTANCE:
                from .calculators.huber_distance import HuberDistanceCalculator

                return HuberDistanceCalculator()
            case TransportDistanceMetric.WASSERSTEIN:
                from .calculators.wasserstein import WassersteinDistanceCalculator

                return WassersteinDistanceCalculator()
        raise ValueError(f"No calculator found for metric {self}")
