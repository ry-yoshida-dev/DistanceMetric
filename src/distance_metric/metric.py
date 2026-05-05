from abc import ABCMeta, abstractmethod
from enum import EnumMeta

from .calculator import DistanceCalculator


class ABCEnumMeta(EnumMeta, ABCMeta):
    """Metaclass that allows abstract Enum groups."""


class DistanceMetric(metaclass=ABCEnumMeta):
    @property
    @abstractmethod
    def calculator(self) -> DistanceCalculator:
        """Return a calculator instance for this metric member."""
