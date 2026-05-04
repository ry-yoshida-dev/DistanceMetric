from enum import Enum
from typing import Type

from ...calculator import DistanceCalculator
from ...metric import DistanceMetric


class GeodesicSequenceDistanceMetric(DistanceMetric, Enum):
    HAVERSINE = "haversine"
    DYNAMIC_TIME_WARPING = "dynamic_time_warping"

    @property
    def calculator(self) -> Type[DistanceCalculator]:
        match self:
            case GeodesicSequenceDistanceMetric.HAVERSINE:
                from .calculators.haversine import HaversineDistanceCalculator

                return HaversineDistanceCalculator
            case GeodesicSequenceDistanceMetric.DYNAMIC_TIME_WARPING:
                from .calculators.dynamic_time_warping import DynamicTimeWarpingDistanceCalculator

                return DynamicTimeWarpingDistanceCalculator
        raise ValueError(f"No calculator found for metric {self}")
