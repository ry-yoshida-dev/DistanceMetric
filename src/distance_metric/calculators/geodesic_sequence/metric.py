"""
Geodesic and sequence metric names (Haversine, Dynamic Time Warping).

Haversine expects latitude-longitude samples; DTW expects sequences along rows.
"""

from enum import Enum

from ...calculator import DistanceCalculator
from ...metric import DistanceMetric


class GeodesicSequenceDistanceMetric(DistanceMetric, Enum):
    """
    Spatial and sequential comparison metrics.

    Members:
    --------
    HAVERSINE : str
        Great-circle distance on a sphere (see radius kwarg).
    DYNAMIC_TIME_WARPING : str
        Alignment cost between two variable-length or misaligned series.

    Notes:
    -----
    DTW cross mode loops over all query and gallery rows and can be costly.
    """

    HAVERSINE = "haversine"
    DYNAMIC_TIME_WARPING = "dynamic_time_warping"

    @property
    def calculator(self) -> DistanceCalculator:
        """
        Calculator instance for this metric.

        Returns:
        --------
        DistanceCalculator
            HaversineDistanceCalculator or DynamicTimeWarpingDistanceCalculator.

        Raises:
        -------
        ValueError
            If the member is not mapped to a calculator.
        """
        match self:
            case GeodesicSequenceDistanceMetric.HAVERSINE:
                from .calculators.haversine import HaversineDistanceCalculator

                return HaversineDistanceCalculator()
            case GeodesicSequenceDistanceMetric.DYNAMIC_TIME_WARPING:
                from .calculators.dynamic_time_warping import DynamicTimeWarpingDistanceCalculator

                return DynamicTimeWarpingDistanceCalculator()
        raise ValueError(f"No calculator found for metric {self}")
