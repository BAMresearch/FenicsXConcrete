import dolfinx as df
import numpy as np


class Sensors(dict):
    """
    Dict that also allows to access the parameter p["parameter"] via the matching attribute p.parameter
    to make access shorter

    When to sensors with the same name are defined, the next one gets a number added to the name
    """

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        assert key in self
        self[key] = value

    def __setitem__(self, initial_key, value):
        # check if key exists, if so, add a number to the name
        i = 2
        key = initial_key
        if key in self:
            while key in self:
                key = initial_key + str(i)
                i += 1

        super().__setitem__(key, value)


# sensor template
class Sensor:
    """Template for a sensor object"""

    def measure(self, problem, t):
        """Needs to be implemented in child, depending on the sensor"""
        raise NotImplementedError()

    @property
    def name(self):
        return self.__class__.__name__

    def data_max(self, value):
        if value > self.max:
            self.max = value