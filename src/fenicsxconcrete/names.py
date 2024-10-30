from pydantic.dataclasses import dataclass
from enum import Enum
from typing import Literal
import pint

FieldVaraibles = Literal["displacement", "temperature", "nonlocal_equivalent_strain"]

QuadratureVariables = Literal["mandel_stress", "mandel_strain", "mandel_strain_rate"]

#@dataclass
#class BaseQuantity:
#    unit: str

#class BaseQuantity(Enum):
#    Displacement = "m"
#    Velocity = "m/s"
#    Acceleration = "m/s^2"
#    Temperature = Quantity("K")
#    Strain = Quantity("1")
#    Stress = Quantity("Pa")
#    Density = Quantity("kg/m^3")
#    Time = Quantity("s")

class SolutionField(Enum):
    Displacement = pint.Unit("m")
    Velocity = pint.Unit("m/s")
    Temperature = pint.Unit("K")
    NonlocalEquivalentStrain = pint.Unit("1")

#@dataclass
#class PDEDefinition:
#    name:
#@dataclass
#class Quantity:
#    name: str
#    value: float
#    unit: str
