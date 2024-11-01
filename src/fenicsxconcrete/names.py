from pydantic.dataclasses import dataclass
from enum import Enum
from typing import Literal
import pint


class SolutionField(Enum):
    Displacement = pint.Unit("m")
    Velocity = pint.Unit("m/s")
    Temperature = pint.Unit("K")
    NonlocalEquivalentStrain = pint.Unit("1")
