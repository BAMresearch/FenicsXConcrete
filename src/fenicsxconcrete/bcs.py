
from pydantic import ConfigDict, RootModel, field_serializer
from pydantic.dataclasses import dataclass
from pydantic.types import conlist
from typing import Annotated, Callable, NewType
import numpy as np
from pint import Unit
from names import SolutionField

import inspect

from pydantic.functional_serializers import PlainSerializer

def marker_serializer(marker: int | np.ndarray | Callable[[np.ndarray], np.ndarray], _info):
    if isinstance(marker, np.ndarray):
        return str(list(marker))
    elif callable(marker):
        return inspect.getsource(marker)
    elif isinstance(marker, int):
        return str(marker)

Marker = Annotated[int | Callable[[np.ndarray], np.ndarray], PlainSerializer(marker_serializer)]


@dataclass(config=dict(arbitrary_types_allowed=True))
class DirichletBCDefinition:
    """
    Definition of a time- and position-dependent Dirichlet boundary condition.
    Note that functions cannot be serialized to JSON!
    """
    marker: Marker
    value: np.ndarray | Callable[[np.ndarray, float], np.ndarray]
    unit: Unit
    subspace: int | None
    variable: SolutionField


@dataclass(config=dict(arbitrary_types_allowed=True))
class NeumannBCDefinition:
    """
    Definition of a time- and position-dependent Neumann boundary condition.
    """
    marker: Marker | Callable[[np.ndarray], np.ndarray]
    value: list[float] | Callable[[np.ndarray, float], np.ndarray] 

@dataclass(config=dict(arbitrary_types_allowed=True))
class BoundaryNormalTermDefinition:
    value: list[float] | Callable[[np.ndarray, float], np.ndarray]
    variable: str

@dataclass(config=dict(arbitrary_types_allowed=True))
class SourceTermDefinition:
    value: list[float] | Callable[[np.ndarray, float], np.ndarray]
    variable: str

@dataclass(config=dict(arbitrary_types_allowed=True))
class InitialConditionDefinition:
    value: list[float] | Callable[[np.ndarray], np.ndarray]
    variable: str
