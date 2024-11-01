
from typing import Callable
from pydantic.dataclasses import dataclass

@dataclass(config=dict(arbitrary_types_allowed=True))
class Material:
    name: str

@dataclass(config=dict(arbitrary_types_allowed=True))
class LinearElasticMaterial(Material):
    mu: float
    lam: float

@dataclass(config=dict(arbitrary_types_allowed=True))
class MisesPlasticityIsotropicHardening(Material):
    mu: float
    lam: float
    yield_stress: float
    hardening_modulus: float | Callable[[float], float]