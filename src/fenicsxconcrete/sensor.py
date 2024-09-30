
from pydantic.dataclasses import dataclass
from typing import Callable
import numpy as np




@dataclass(config=dict(arbitrary_types_allowed=True))
class PointSensorDefinition:
    """
    Definition of a point sensor

    Args:
        location: The location of the sensor
        variable: The variable to be measured
     
    """
    location: tuple[float,float,float]
    variable: str
    unit: str

@dataclass(config=dict(arbitrary_types_allowed=True))
class GlobalSensorDefinition:
    variable: str
    unit: str

@dataclass(config=dict(arbitrary_types_allowed=True))
class PlotDefinition:
    variable: str
    #mapping: Callable
    unit: str

# @dataclass
# class DolfinXPointSensor:
#     cells: list[int]
#     plot_function: df.fem.Function | None
#     function: df.fem.Function
#     mapping: Callable
#     definition: PointSensorDefinition

#     def __init__(self, sensor: SensorDefinition, function: df.fem.FunctionSpace, plot_function: df.fem.Function | None = None):
#         pass

@dataclass
class  Sensors:
    groups: dict[str, list[PointSensorDefinition | GlobalSensorDefinition | PlotDefinition]]
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
                    case PointSensorDefinition(location, variable, mapping, unit):
                        pass
                    case GlobalSensorDefinition(variable, mapping, unit):
                        pass
                    case PlotDefinition(variable, mapping, unit):
                        pass
                    case _:
                        pass