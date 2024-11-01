from typing import ClassVar, Literal
import numpy as np
from pydantic import Field, RootModel
from bcs import DisplacementBC, ForceBC, TemperatureBC, PressureBC, BodyForce, VolumetricHeatFlux, InitialCondition
from mesh import MeshGenerator
from material import MaterialDefinition, LinearElasticMaterial
from pydantic.dataclasses import dataclass

@dataclass(config=dict(arbitrary_types_allowed=True))
class Experiment:
    name: ClassVar[str] = "experiment"
    initial_conditions: list[InitialCondition] | None
    geometry: MeshGenerator
    time: tuple[float, float]
    material: MaterialDefinition

@dataclass(config=dict(arbitrary_types_allowed=True))
class MechanicsExperiment(Experiment):
    displacement_bcs: list[DisplacementBC] | None
    pressure_bcs: list[PressureBC] | None
    force_bcs: list[ForceBC] | None
    body_forces: list[BodyForce] | None


@dataclass(config=dict(arbitrary_types_allowed=True))
class HeatTransferExperiment(Experiment):
    temperature_bcs: list[TemperatureBC] | None
    heat_flux_bcs: list[VolumetricHeatFlux] | None
    heat_flux: list[VolumetricHeatFlux] | None


@dataclass(config=dict(arbitrary_types_allowed=True))
class ThermoMechanicalExperiment(MechanicsExperiment, HeatTransferExperiment):
    pass


if __name__ == "__main__":
    exp = ThermoMechanicalExperiment(
        displacement_bcs=None,
        pressure_bcs=None,
        force_bcs=None,
        initial_conditions=None,
        body_forces=None,
        geometry=MeshGenerator(
            parameters={"length": (1, "m")}, mesh_tags={"left": 0, "right": 1, "top": 2, "bottom": 3}
        ),
        time=(0.0, 1.0),
        material=LinearElasticMaterial(name="steel", mu=1.0, lam=2.0),
        temperature_bcs=None,
        heat_flux_bcs=None,
        heat_flux=None,
    )
    print(RootModel[ThermoMechanicalExperiment](exp).model_dump_json(indent=4))
