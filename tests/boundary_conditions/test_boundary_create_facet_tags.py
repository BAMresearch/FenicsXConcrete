import dolfinx
from mpi4py import MPI

from fenicsxconcrete.boundary_conditions.boundary import create_facet_tags, plane_at


def test_create_facet_tags() -> None:
    domain = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, 8, 8, dolfinx.mesh.CellType.triangle
    )
    to_be_marked = {"bottom": (4, plane_at(0.0, "y")), "right": (5, plane_at(1.0, "x"))}
    ft, marked = create_facet_tags(domain, to_be_marked)
    ft_bottom = ft.find(4)
    ft_right = ft.find(5)
    assert ft_bottom.size == 8
    assert ft_right.size == 8
    assert "bottom" in marked.keys()
    assert "right" in marked.keys()
    assert 4 in marked.values()
    assert 5 in marked.values()


if __name__ == "__main__":
    test_create_facet_tags()
