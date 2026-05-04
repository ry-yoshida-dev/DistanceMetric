from enum import Enum
from typing import Type

from ...calculator import DistanceCalculator
from ...metric import DistanceMetric


class RatioBasedDistanceMetric(DistanceMetric, Enum):
    CANBERRA = "canberra"
    BRAY_CURTIS = "bray_curtis"

    @property
    def calculator(self) -> Type[DistanceCalculator]:
        match self:
            case RatioBasedDistanceMetric.CANBERRA:
                from .calculators.canberra import CanberraDistanceCalculator

                return CanberraDistanceCalculator
            case RatioBasedDistanceMetric.BRAY_CURTIS:
                from .calculators.bray_curtis import BrayCurtisDistanceCalculator

                return BrayCurtisDistanceCalculator
        raise ValueError(f"No calculator found for metric {self}")
