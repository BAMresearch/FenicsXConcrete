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
        self.units = self.base_unit()

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
        return self.data * self.units

    def get_time_list(self):
        """Returns the time data with respective unit"""
        return self.time * ureg.second

    def get_data_at_time(self, t):
        """Returns the measured data at a specific time"""
        try:
            i = self.time.index(t)
        except ValueError:  # I want my own value error that is meaningful to the input
            raise ValueError(f"There is no data measured at time {t}")

        return self.data[i] * self.units

    def get_last_data_point(self):
        """Returns the measured data with respective unit"""
        return self.data[-1] * self.units


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
