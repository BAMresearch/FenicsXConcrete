"""Based on Philipp Diercks implementation for multi"""

import dolfinx
import numpy as np
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from fenicsxconcrete.boundary_conditions.boundary import plane_at


def test_square() -> None:
    domain = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, 8, 8, dolfinx.mesh.CellType.quadrilateral
    )
    V = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 1))

    bottom = plane_at(0.0, "y")
    top = plane_at(1.0, "y")
    left = plane_at(0.0, "x")
    right = plane_at(1.0, "x")

    dofs = dolfinx.fem.locate_dofs_geometrical(V, bottom)
    bc = dolfinx.fem.dirichletbc(ScalarType(42), dofs, V)
    ndofs = bc.dof_indices()[1]
    assert ndofs == 9
    assert bc.g.value == 42

    def on_boundary(x):
        return np.logical_or(top(x), bottom(x))

    dofs = dolfinx.fem.locate_dofs_geometrical(V, on_boundary)
    bc = dolfinx.fem.dirichletbc(ScalarType(17), dofs, V)
    ndofs = bc.dof_indices()[1]
    assert ndofs == 9 * 2
    assert bc.g.value == 17

    def origin(x):
        return np.logical_and(left(x), bottom(x))

    dofs = dolfinx.fem.locate_dofs_geometrical(V, origin)
    bc = dolfinx.fem.dirichletbc(ScalarType(21), dofs, V)
    ndofs = bc.dof_indices()[1]
    assert ndofs == 1
    assert bc.g.value == 21

    def l_shaped_boundary(x):
        return np.logical_or(top(x), right(x))

    facet_dim = 1
    boundary_facets = dolfinx.mesh.locate_entities_boundary(
        domain, facet_dim, l_shaped_boundary
    )
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, facet_dim, boundary_facets)
    bc = dolfinx.fem.dirichletbc(ScalarType(666), boundary_dofs, V)
    ndofs = bc.dof_indices()[1]
    assert ndofs == 17
    assert bc.g.value == 666


def test_cube() -> None:
    nx, ny, nz = 4, 4, 4
    domain = dolfinx.mesh.create_unit_cube(
        MPI.COMM_WORLD, nx, ny, nz, dolfinx.mesh.CellType.hexahedron
    )
    V = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 1))

    xy_plane = plane_at(0.0, "z")
    dofs = dolfinx.fem.locate_dofs_geometrical(V, xy_plane)
    nodal_value = 42
    bc = dolfinx.fem.dirichletbc(ScalarType(nodal_value), dofs, V)
    ndofs = bc.dof_indices()[1]
    assert ndofs == (nx + 1) * (ny + 1)
    assert bc.g.value == nodal_value


if __name__ == "__main__":
    test_square()
    test_cube()
