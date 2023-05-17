from __future__ import annotations

from abc import ABC, abstractmethod

import pint

from fenicsxconcrete.util import LogMixin, ureg


# sensor template
class BaseSensor(ABC, LogMixin):
    """Template for a sensor object

    Attributes:
        data: list of measured values
        time: list of time stamps
        units : pint definition of the base unit a sensor returns
        name : name of the sensor, default is class name, but can be changed
    """

    def __init__(self, name: str | None = None) -> None:
        """initializes the sensor

        Args:
            name: optional argument to set a specific sensor name
        """
        self.data = []
        self.time = []
        self.units = self.base_unit()
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name

    @abstractmethod
    def measure(self):
        """Needs to be implemented in child, depends on the sensor

        This function is called, when the sensor adds the data to the data list.
        """

    @staticmethod
    @abstractmethod
    def base_unit():
        """Defines the base unit of this sensor"""

    def get_data_list(self) -> pint.Quantity[list]:
        """Returns the measured data with respective unit

        Returns:
            measured data list with respective unit
        """
        data = self.data * self.base_unit()  # add base units
        data.ito(self.units)  # convert to target units
        return data

    def get_time_list(self) -> pint.Quantity[list]:
        """Returns the time data with respective unit

        Returns:
            the time stamp list with the respective unit
        """
        return self.time * ureg.second

    def get_data_at_time(self, t: float) -> pint.Quantity:
        """Returns the measured data at a specific time

        Returns:
            measured data at the specified time with the unit

        Raises:
            ValueError: If there is no value t in time list
        """
        try:
            i = self.time.index(t)
        except ValueError:  # I want my own value error that is meaningful to the input
            raise ValueError(f"There is no data measured at time {t}")

        data = self.data[i] * self.base_unit()  # add base units
        data.ito(self.units)  # convert to target units

        return data

    def get_last_entry(self) -> pint.Quantity:
        """Returns the measured data with respective unit

        Returns:
            the measured data list with the respective unit

        Raises:
            RuntimeError: If the data list is empty
        """
        if len(self.data) > 0:
            data = self.data[-1] * self.base_unit()  # add base units
            data.ito(self.units)  # convert to target units
            return data
        else:
            raise RuntimeError("There is no measured data to retrieve.")

    def set_units(self, units: str) -> None:
        """sets the units which the sensor should return

        the unit must match the dimensionality of the base unit

        Args:
            units: name of the units to convert to, must be defined in pint unit registry
        """
        new_unit = ureg(units)
        assert self.base_unit().dimensionality == new_unit.dimensionality
        self.units = new_unit


class PointSensor(BaseSensor):
    """
    Abstract class for a sensor that measures values at a specific point

    Attributes:
        data: list of measured values
        time: list of time stamps
        units : pint definition of the base unit a sensor returns
        name : name of the sensor, default is class name, but can be changed
        where: location where the value is measured
    """

    def __init__(self, where: list[int | float], name: str | None = None) -> None:
        """
        initializes a point sensor, for further details, see base class

        Arguments:
            where : Point where to measure
            name : name of the sensor
        """
        super().__init__(name=name)
        self.where = where

    @abstractmethod
    def measure(self):
        """Needs to be implemented in child, depending on the sensor"""

    @staticmethod
    @abstractmethod
    def base_unit():
        """Defines the base unit of this sensor, must be specified by child"""
