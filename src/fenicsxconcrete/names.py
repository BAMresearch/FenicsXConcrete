from pydantic.dataclasses import dataclass
from enum import Enum
from typing import Literal

FieldVaraibles = Literal["displacement", "temperature", "nonlocal_equivalent_strain"]

QuadratureVariables = Literal["mandel_stress", "mandel_strain", "mandel_strain_rate"]

@dataclass
class Quantity:
    unit: str

class BaseQuantity(Enum):
    Displacement = Quantity("m")
    Velocity = Quantity("m/s")
    Acceleration = Quantity("m/s^2")
    Temperature = Quantity("K")
    Strain = Quantity("1")
    Stress = Quantity("Pa")
    Density = Quantity("kg/m^3")
    Time = Quantity("s")

class SolutionFields(Enum):
    Displacement = Quantity("m")
    Temperature = Quantity("K")
    NonlocalEquivalentStrain = Quantity("1")
