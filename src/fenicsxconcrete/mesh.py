from pint import Quantity
from pydantic.dataclasses import dataclass
@dataclass(config=dict(arbitrary_types_allowed=True))
class MeshGenerator:
    def generate(degree: int):
        pass

    def cell_tags(self) -> dict[str, int]:
        pass

    def facet_tags(self) -> dict[str, int]:
        pass

    def material_tags(self) -> dict[str, int]:
        pass

@dataclass(config=dict(arbitrary_types_allowed=True))
class UnitCubeMesh(MeshGenerator):
    point0: Quantity
    point1: Quantity

    def generate(self, degree: int):
        pass

@dataclass(config=dict(arbitrary_types_allowed=True))
class PlateWithHole(MeshGenerator):
    radius: Quantity
    length: Quantity
    width: Quantity

    def generate(self, degree: int):
        pass

@dataclass(config=dict(arbitrary_types_allowed=True))
class CantileverBeam(MeshGenerator):
    length: Quantity
    height: Quantity
    width: Quantity

    def generate(self, degree: int):
        pass