from enum import Enum
from typing import Type

from ...calculator import DistanceCalculator
from ...metric import DistanceMetric


class InformationTheoreticDistanceMetric(DistanceMetric, Enum):
    KL_DIVERGENCE = "kl_divergence"
    JENSEN_SHANNON_DIVERGENCE = "jensen_shannon_divergence"
    BHATTACHARYYA = "bhattacharyya"
    HELLINGER = "hellinger"

    @property
    def calculator(self) -> Type[DistanceCalculator]:
        match self:
            case InformationTheoreticDistanceMetric.KL_DIVERGENCE:
                from .calculators.kl_divergence import KLDivergenceDistanceCalculator

                return KLDivergenceDistanceCalculator
            case InformationTheoreticDistanceMetric.JENSEN_SHANNON_DIVERGENCE:
                from .calculators.jensen_shannon_divergence import (
                    JensenShannonDivergenceDistanceCalculator,
                )

                return JensenShannonDivergenceDistanceCalculator
            case InformationTheoreticDistanceMetric.BHATTACHARYYA:
                from .calculators.bhattacharyya import BhattacharyyaDistanceCalculator

                return BhattacharyyaDistanceCalculator
            case InformationTheoreticDistanceMetric.HELLINGER:
                from .calculators.hellinger import HellingerDistanceCalculator

                return HellingerDistanceCalculator
        raise ValueError(f"No calculator found for metric {self}")
