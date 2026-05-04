from enum import Enum
from typing import Type

from ...calculator import DistanceCalculator
from ...metric import DistanceMetric


class RobustTransportDistanceMetric(DistanceMetric, Enum):
    HUBER_DISTANCE = "huber_distance"
    WASSERSTEIN = "wasserstein"

    @property
    def calculator(self) -> Type[DistanceCalculator]:
        match self:
            case RobustTransportDistanceMetric.HUBER_DISTANCE:
                from .calculators.huber_distance import HuberDistanceCalculator

                return HuberDistanceCalculator
            case RobustTransportDistanceMetric.WASSERSTEIN:
                from .calculators.wasserstein import WassersteinDistanceCalculator

                return WassersteinDistanceCalculator
        raise ValueError(f"No calculator found for metric {self}")
