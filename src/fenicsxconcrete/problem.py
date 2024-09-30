from pydantic.dataclasses import dataclass

from pydantic import RootModel

from bcs import (
    BodyForceDefinition,
    DirichletBCDefinition,
    InitialConditionDefinition,
    NeumannBCDefinition,
)
from material import LinearElasticMaterial
from mesh import MeshGenerator

from sensor import PlotDefinition, PointSensorDefinition, GlobalSensorDefinition
from experiment import Experiment

@dataclass(config=dict(arbitrary_types_allowed=True))
class QuadratureRuleDefinition:
    name: str
    order: int
    variable: str

@dataclass(config=dict(arbitrary_types_allowed=True))
class FiniteElementDefinition:
    name: str
    geometry_order: int
    function_order: int
    variable: str

@dataclass(config=dict(arbitrary_types_allowed=True))
class FEMProblemDefinition:
    experiment: Experiment
    sensors: list[PointSensorDefinition | GlobalSensorDefinition]
    plots: list[PlotDefinition]
    element_type: list[FiniteElementDefinition]
    quadrature_rule: list[QuadratureRuleDefinition]


if __name__ == "__main__":
    bc = DirichletBCDefinition(marker=1, value=[0.0, 0.0], subspace=0, variable="displacement")
    neumann = NeumannBCDefinition(marker=2, value=[-42.0], variable="displacement")
    initial = InitialConditionDefinition(value=[42.24], variable="density")
    body_force = BodyForceDefinition(value=[0.0, 0.0, 9.81], variable="displacement")
    mat = LinearElasticMaterial(name="steel", mu=1.0, lam=2.0)
    geo = MeshGenerator(parameters={"length": (1, "m")}, mesh_tags={"left": 0, "right": 1, "top": 2, "bottom": 3})
    solution_fields = ["displacement"]
    time = (0.0, 1.0)
    exp = Experiment(
        dirichlet_bcs=[bc],
        neumann_bcs=[neumann],
        initial_conditions=[initial],
        body_forces=[body_force],
        geometry=geo,
        solution_fields=solution_fields,
        time=time,
        material=mat,
    )

    sensor = PointSensorDefinition(location=(0.0, 0.0, 0.0), variable="displacement", unit="m")
    global_sensor = GlobalSensorDefinition(variable="energy", unit="J")

    plot = PlotDefinition(variable="displacement", unit="m")

    problem = FEMProblemDefinition(
        experiment=exp,
        sensors=[sensor, global_sensor],
        plots=[plot],
        element_type=[FiniteElementDefinition(name="P", geometry_order=1, function_order=1, variable="displacement")],
        quadrature_rule=[QuadratureRuleDefinition(name="Gauss", order=2, variable="displacement")],
    )

    print(RootModel[Experiment](exp).model_dump_json(indent=4))
    print(RootModel[FEMProblemDefinition](problem).model_dump_json(indent=4, serialize_as_any=True))
