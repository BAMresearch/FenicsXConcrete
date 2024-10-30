from typing import Literal
import numpy as np
from pydantic import Field, RootModel
from bcs import DirichletBCDefinition, InitialConditionDefinition, NeumannBCDefinition, BodyForceDefinition
from mesh import MeshGenerator
from material import MaterialDefinition, LinearElasticMaterial
from dataclasses import dataclass, asdict

@dataclass#(config=dict(arbitrary_types_allowed=True))
class Experiment:
    dirichlet_bcs: list[DirichletBCDefinition] | None
    neumann_bcs: list[NeumannBCDefinition] | None
    initial_conditions: list[InitialConditionDefinition] | None
    body_forces: list[BodyForceDefinition] | None
    geometry: MeshGenerator
    solution_fields: list[str]
    time: tuple[float, float]
    material: MaterialDefinition
    name: str ='experiment'




if __name__ == "__main__":
    def marker_function(x):
        return np.isclose(x[1], 0)
    bc = DirichletBCDefinition(marker=marker_function, value=np.array([0.,0.]), subspace=0, variable='displacement')
    neumann = NeumannBCDefinition(marker=2, value=[-42.0], variable='displacement')
    initial = InitialConditionDefinition(value=[42.24], variable='density')
    body_force = BodyForceDefinition(value=[0.,0., 9.81], variable='displacement')
    mat = LinearElasticMaterial(name='steel', mu=1., lam=2.)
    geo = MeshGenerator(parameters={'length': (1, 'm')}, mesh_tags={'left': 0, 'right': 1, 'top': 2, 'bottom': 3})
    solution_fields = ['displacement']
    time = (0., 1.)
    exp = Experiment(dirichlet_bcs=[bc], neumann_bcs=[neumann], initial_conditions=[initial], body_forces=[body_force], geometry=geo, solution_fields=solution_fields, time=time, material=mat)

    print(asdict(exp))
    print(RootModel[Experiment](exp).model_dump_json(indent=4))
    
