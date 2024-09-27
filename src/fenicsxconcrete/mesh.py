from pydantic.dataclasses import dataclass
@dataclass(config=dict(arbitrary_types_allowed=True))
class MeshGenerator:
    parameters: dict[str, tuple[float, str]] | None
    cell_tags: dict[str, int] | None = None
    facet_tags: dict[str, int] | None = None

    def generate(degree: int):
        pass