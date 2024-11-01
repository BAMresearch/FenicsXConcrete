
from names import SolutionField
from pydantic import ConfigDict, RootModel, field_serializer
from pydantic.dataclasses import dataclass
from pydantic.types import conlist
from typing import Annotated, Callable, ClassVar, NewType
import numpy as np
from pint import Unit
#from names import SolutionField

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
Unit = Annotated[Unit, PlainSerializer(lambda unit, _info: str(unit))]

@dataclass(config=dict(arbitrary_types_allowed=True))
class DisplacementBC:
    """
    Definition of a time- and position-dependent displacement Dirichlet boundary condition.
    Note that functions cannot be serialized to JSON!
    """
    name: ClassVar[str] = "displacement_bc"
    marker: Marker
    value: np.ndarray | Callable[[np.ndarray, float], np.ndarray]
    unit: Unit
    subspace: int | None


@dataclass(config=dict(arbitrary_types_allowed=True))
class TemperatureBC:
    """
    Definition of a time- and position-dependent temperature Dirichlet boundary condition.
    Note that functions cannot be serialized to JSON!
    """
    name: ClassVar[str] = "temperature_bc"
    marker: Marker
    value: float | Callable[[np.ndarray, float], float]
    unit: Unit

@dataclass(config=dict(arbitrary_types_allowed=True))
class PressureBC:
    """
    Definition of a time- and position-dependent pressure Neumann boundary condition.
    Note that functions cannot be serialized to JSON!
    """
    name: ClassVar[str] = "pressure_bc"
    marker: Marker
    value: float | Callable[[np.ndarray, float], float]
    unit: Unit

@dataclass(config=dict(arbitrary_types_allowed=True))
class ForceBC:
    """
    Definition of a time- and position-dependent force Neumann boundary condition.
    Note that functions cannot be serialized to JSON!
    """
    name: ClassVar[str] = "force_bc"
    marker: Marker
    value: float | Callable[[np.ndarray, float], float]
    unit: Unit

@dataclass(config=dict(arbitrary_types_allowed=True))
class HeatFluxBC:
    """
    Definition of a time- and position-dependent heat flux Neumann boundary condition.
    Note that functions cannot be serialized to JSON!
    """
    name: ClassVar[str] = "heat_flux_bc"
    marker: Marker
    value: np.ndarray | Callable[[np.ndarray, float], np.ndarray]
    unit: Unit

@dataclass(config=dict(arbitrary_types_allowed=True))
class BodyForce:
    """
    Definition of a time- and position-dependent body force.
    Note that functions cannot be serialized to JSON!
    """
    name: ClassVar[str] = "body_force"
    value: np.ndarray | Callable[[np.ndarray, float], np.ndarray]
    unit: Unit

@dataclass(config=dict(arbitrary_types_allowed=True))
class VolumetricHeatFlux:
    """
    Definition of a time- and position-dependent volumetric heat flux.
    Note that functions cannot be serialized to JSON!
    """
    name: ClassVar[str] = "volumetric_heat_flux"
    value: float | Callable[[np.ndarray, float], float]
    unit: Unit

@dataclass(config=dict(arbitrary_types_allowed=True))
class InitialCondition:
    """
    Definition of a time- and position-dependent initial condition.
    Note that functions cannot be serialized to JSON!
    """
    name: ClassVar[str] = "initial_condition"
    value: float | Callable[[np.ndarray], float]
    field: str | SolutionField
    unit: Unit

if __name__ == "__main__":
    bc = DisplacementBC(marker=2, value=np.array([0.,0.]), unit=Unit('m'), subspace=0)
    ic = InitialCondition(value=42.24, field='density', unit=Unit('kg/m^3'))
    print(RootModel[InitialCondition](ic).model_dump_json(indent=4))