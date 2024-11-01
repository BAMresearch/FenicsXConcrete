from pydantic.dataclasses import dataclass

from pydantic import RootModel

from bcs import *
from material import LinearElasticMaterial
from mesh import MeshGenerator

from sensor import Plot, PointSensor, GlobalSensor
from experiment import Experiment

@dataclass(config=dict(arbitrary_types_allowed=True))
class QuadratureRule:
    name: str
    order: int
    variable: str

@dataclass(config=dict(arbitrary_types_allowed=True))
class FiniteElement:
    name: str
    geometry_order: int
    function_order: int
    variable: str

@dataclass(config=dict(arbitrary_types_allowed=True))
class FEMProblem:
    experiment: Experiment
    sensors: list[PointSensor | GlobalSensor]
    plots: list[Plot]
    element_type: list[FiniteElement]
    quadrature_rule: list[QuadratureRule]

    def solve(self):
        pass

    def step(self, t: float):
        pass

    def measure_and_plot(self):
        pass

    def measure(self):
        pass

    def sensors_as_pandas(self):
        pass



if __name__ == "__main__":
    pass