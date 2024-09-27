
from pydantic import ConfigDict, RootModel
from pydantic.dataclasses import dataclass
from pydantic.types import conlist
from typing import Callable, NewType
import numpy as np

Marker = NewType('Marker', int)

@dataclass(config=dict(arbitrary_types_allowed=True))
class DirichletBCDefinition:
    """
    Definition of a time- and position-dependent Dirichlet boundary condition.
    Note that functions cannot be serialized to JSON!
    """
    marker: Marker | list[float] | Callable[[np.ndarray], np.ndarray]
    value: conlist(float, min_length=1, max_length=3) | Callable[[np.ndarray, float], np.ndarray]
    subspace: int | None
    variable: str

@dataclass(config=dict(arbitrary_types_allowed=True))
class NeumannBCDefinition:
    """
    Definition of a time- and position-dependent Neumann boundary condition.
    """
    marker: Marker
    value: conlist(float, min_length=1, max_length=3) | Callable[[np.ndarray, float], np.ndarray] 

@dataclass(config=dict(arbitrary_types_allowed=True))
class BodyForceDefinition:
    value: conlist(float, min_length=1, max_length=3)

@dataclass(config=dict(arbitrary_types_allowed=True))
class InitialConditionDefinition:
    value: list[float]
    variable: str
