from enum import Enum
from typing import Type

from ...calculator import DistanceCalculator
from ...metric import DistanceMetric


class SetMismatchDistanceMetric(DistanceMetric, Enum):
    HAMMING = "hamming"
    JACCARD = "jaccard"

    @property
    def calculator(self) -> Type[DistanceCalculator]:
        match self:
            case SetMismatchDistanceMetric.HAMMING:
                from .calculators.hamming import HammingDistanceCalculator

                return HammingDistanceCalculator
            case SetMismatchDistanceMetric.JACCARD:
                from .calculators.jaccard import JaccardDistanceCalculator

                return JaccardDistanceCalculator
        raise ValueError(f"No calculator found for metric {self}")
