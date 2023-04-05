from __future__ import annotations

from abc import ABC, abstractmethod

from fenicsxconcrete.helper import LogMixin
from fenicsxconcrete.unit_registry import ureg


# sensor template
class BaseSensor(ABC, LogMixin):
    """Template for a sensor object"""

    def __init__(self) -> None:
        self.data = []
        self.time = []
        self.unit = self.base_unit()

    @abstractmethod
    def measure(self):
        """Needs to be implemented in child, depending on the sensor"""

    @staticmethod
    @abstractmethod
    def base_unit():
        """Defines the base unit of this sensor"""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def get_data_list(self):
        """Returns the measured data with respective unit"""
        return self.data * self.unit

    def get_time_list(self):
        """Returns the time data with respective unit"""
        return self.data * ureg.second

    def get_data_at_time(self, t):
        """Returns the measured data at a specific time"""
        try:
            i = self.time.index(t)
        except ValueError:  # I want my own value error that is meaningful to the input
            raise ValueError(f"There is no data measured at time {t}")

        return self.data[i] * self.unit


class PointSensor(BaseSensor):
    """A sensor that measures values at a specific point"""

    def __init__(self, where: list[int | float]) -> None:
        """
        Arguments:
            where : Point where to measure
        """
        super().__init__()
        self.where = where

    @abstractmethod
    def measure(self):
        """Needs to be implemented in child, depending on the sensor"""

    @staticmethod
    @abstractmethod
    def base_unit():
        """Defines the base unit of this sensor"""
