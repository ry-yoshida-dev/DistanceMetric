"""
Information-theoretic divergence metric names (KL, JS, Bhattacharyya, Hellinger).

Treats vectors as nonnegative mass over aligned discrete bins.
"""

from enum import Enum

from ...calculator import DistanceCalculator
from ...metric import DistanceMetric


class InformationTheoreticDistanceMetric(DistanceMetric, Enum):
    """
    Divergences and distances derived from probability-like mass vectors.

    Members:
    --------
    KL_DIVERGENCE, JENSEN_SHANNON_DIVERGENCE, BHATTACHARYYA, HELLINGER
        Each string value selects the matching calculator under calculators/.

    Notes:
    -----
    KL divergence is asymmetric in query versus gallery roles.
    """

    KL_DIVERGENCE = "kl_divergence"
    JENSEN_SHANNON_DIVERGENCE = "jensen_shannon_divergence"
    BHATTACHARYYA = "bhattacharyya"
    HELLINGER = "hellinger"

    @property
    def calculator(self) -> DistanceCalculator:
        """
        Calculator instance for this metric.

        Returns:
        --------
        DistanceCalculator
            One of the divergence calculators in calculators/.

        Raises:
        -------
        ValueError
            If the member is not mapped to a calculator.
        """
        match self:
            case InformationTheoreticDistanceMetric.KL_DIVERGENCE:
                from .calculators.kl_divergence import KLDivergenceDistanceCalculator

                return KLDivergenceDistanceCalculator()
            case InformationTheoreticDistanceMetric.JENSEN_SHANNON_DIVERGENCE:
                from .calculators.jensen_shannon_divergence import (
                    JensenShannonDivergenceDistanceCalculator,
                )

                return JensenShannonDivergenceDistanceCalculator()
            case InformationTheoreticDistanceMetric.BHATTACHARYYA:
                from .calculators.bhattacharyya import BhattacharyyaDistanceCalculator

                return BhattacharyyaDistanceCalculator()
            case InformationTheoreticDistanceMetric.HELLINGER:
                from .calculators.hellinger import HellingerDistanceCalculator

                return HellingerDistanceCalculator()
        raise ValueError(f"No calculator found for metric {self}")
