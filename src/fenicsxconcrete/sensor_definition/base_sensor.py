from __future__ import annotations

import typing
from abc import ABC, abstractmethod

from fenicsxconcrete.helper import LogMixin

if typing.TYPE_CHECKING:
    from fenicsxconcrete.finite_element_problem.base_material import MaterialProblem


# sensor template
class BaseSensor(ABC, LogMixin):
    """Template for a sensor object"""

    @abstractmethod
    def measure(self, problem: MaterialProblem, t: float):
        """Needs to be implemented in child, depending on the sensor"""
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def data_max(self, value: float) -> None:
        if value > self.max:
            self.max = value
