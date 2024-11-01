
from pydantic.dataclasses import dataclass
from typing import Callable
import numpy as np
from names import SolutionField
from bcs import Unit


@dataclass(config=dict(arbitrary_types_allowed=True))
class PointSensor:
    """
    Definition of a point sensor

    Args:
        location: The location of the sensor
        variable: The variable to be measured
     
    """
    location: tuple[float,float,float]
    variable: str | SolutionField
    unit: Unit

@dataclass(config=dict(arbitrary_types_allowed=True))
class GlobalSensor:
    variable: str | SolutionField
    unit: Unit

@dataclass(config=dict(arbitrary_types_allowed=True))
class Plot:
    variable: str | SolutionField
    unit: Unit


@dataclass
class  Sensors:
    groups: dict[str | SolutionField, list[PointSensor | GlobalSensor | Plot]]
    #plot_functions: dict[str, df.fem.Function | None]
    #functions: dict[str, df.fem.Function]

    def measure(self, t: float):
        for plot_function, function in zip(self.plot_functions.values(), self.functions.values()):
            if plot_function is not None:
                # projection here for all sensors that use the same variable
                # FEniCS code comes here
                pass 
        for group, sensors in self.groups.items():
            for sensor in sensors:
                match sensor:
                    case PointSensor(location, variable, mapping, unit):
                        pass
                    case GlobalSensor(variable, mapping, unit):
                        pass
                    case Plot(variable, mapping, unit):
                        pass
                    case _:
                        pass