"""Based on Philipp Diercks implementation for multi"""

import dolfinx
import numpy as np
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from fenicsxconcrete.boundary_conditions.boundary import line_at, plane_at


def test_cube() -> None:
    n = 4
    domain = dolfinx.mesh.create_unit_cube(
        MPI.COMM_WORLD, n, n, n, dolfinx.mesh.CellType.hexahedron
    )
    V = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 1))

    x_axis = line_at([0, 0], ["z", "y"])
    y_axis = line_at([0, 0], ["z", "x"])
    z_axis = line_at([0, 0], ["x", "y"])

    axis_list = [x_axis, y_axis, z_axis]

    for axis in axis_list:
        dofs = dolfinx.fem.locate_dofs_geometrical(V, x_axis)
        nodal_value = 42
        bc = dolfinx.fem.dirichletbc(ScalarType(nodal_value), dofs, V)
        ndofs = bc.dof_indices()[1]
        assert ndofs == (n + 1)
        assert bc.g.value == nodal_value
