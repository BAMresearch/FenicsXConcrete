
from typing import Callable
from pydantic.dataclasses import dataclass

@dataclass(config=dict(arbitrary_types_allowed=True))
class MaterialDefinition:
    name: str

@dataclass(config=dict(arbitrary_types_allowed=True))
class LinearElasticMaterial(MaterialDefinition):
    mu: float
    lam: float

@dataclass(config=dict(arbitrary_types_allowed=True))
class MisesPlasticityIsotropicHardening(MaterialDefinition):
    mu: float
    lam: float
    yield_stress: float
    hardening_modulus: float | Callable[[float], float]