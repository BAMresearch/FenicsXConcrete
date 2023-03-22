"""Based on Philipp Diercks implementation for multi"""

import dolfinx
import numpy as np
from mpi4py import MPI

from fenicsxconcrete.boundary_conditions.bcs import (
    BoundaryConditions,
    get_boundary_dofs,
)
from fenicsxconcrete.boundary_conditions.boundary import plane_at


def num_square_boundary_dofs(n: int, deg: int, dim: int, num_edges: int = 4) -> int:
    """returns number of dofs for a square
    assumes quadrilateral cells and structured grid

    there are (n+1) * num_edges - num_edges points
    and if degree == 2: additional n*num_edges dofs (edges)
    thus n * num_edges * degree dofs for degree in (1, 2)
    times number of components (i.e. dim)
    """
    return num_edges * n * deg * dim


def num_square_dofs(ncells: int, deg: int, dim: int) -> int:
    if deg == 1:
        n = ncells + 1
    elif deg == 2:
        n = 2 * ncells + 1
    return n**2 * dim


def test_whole_boundary() -> None:
    """test for bcs on ∂Ω

    compare options
        (a) usage of BoundaryConditions
        (b) helper function `get_boundary_dofs`

    The difference is that in (a) it is done topologically,
    whereas in (b) the input is *any* callable (geometrical marker).
    """

    n = 8
    degree = 2
    dim = 2

    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n, dolfinx.mesh.CellType.quadrilateral)
    V = dolfinx.fem.VectorFunctionSpace(domain, ("Lagrange", degree), dim=dim)

    # option (a)
    bc_handler = BoundaryConditions(domain, V)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
    u = dolfinx.fem.Function(V)
    u.x.set(0.0)
    bc_handler.add_dirichlet_bc(u, boundary_facets, method="topological", entity_dim=1)
    bcs = bc_handler.bcs
    dofs = bcs[0].dof_indices()[0]
    assert dofs.size == num_square_boundary_dofs(n, degree, dim)

    def everywhere(x):
        return np.full(x[0].shape, True, dtype=bool)

    # option (b)
    dofs = get_boundary_dofs(V, everywhere)
    assert dofs.size == num_square_boundary_dofs(n, degree, dim)


def test_xy_plane() -> None:
    n = 4
    degree = 2
    dim = 3

    domain = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n, dolfinx.mesh.CellType.hexahedron)
    V = dolfinx.fem.VectorFunctionSpace(domain, ("Lagrange", degree), dim=dim)
    xy_plane = plane_at(0.0, "z")

    # option (a)
    bc_handler = BoundaryConditions(domain, V)
    u = dolfinx.fem.Function(V)
    u.x.set(0.0)
    bc_handler.add_dirichlet_bc(u, xy_plane, method="geometrical")
    bcs = bc_handler.bcs
    dofs = bcs[0].dof_indices()[0]
    assert dofs.size == num_square_dofs(n, degree, dim)

    # option (b)
    dofs = get_boundary_dofs(V, xy_plane)
    assert dofs.size == num_square_dofs(n, degree, dim)


if __name__ == "__main__":
    test_whole_boundary()
    test_xy_plane()
