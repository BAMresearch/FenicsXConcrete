from __future__ import annotations

from abc import ABC, abstractmethod

from fenicsxconcrete.helper import LogMixin


# sensor template
class BaseSensor(ABC, LogMixin):
    """Template for a sensor object"""

    @abstractmethod
    def measure(self):
        """Needs to be implemented in child, depending on the sensor"""

    @property
    def name(self) -> str:
        return self.__class__.__name__
