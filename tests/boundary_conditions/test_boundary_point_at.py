"""Based on Philipp Diercks implementation for multi"""

import dolfinx
import numpy as np
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from fenicsxconcrete.boundary_conditions.boundary import point_at


def test_type_error() -> None:
    """test TypeError in conversion to float"""
    n = 10
    domain = dolfinx.mesh.create_interval(MPI.COMM_WORLD, n, [0.0, 10.0])
    V = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 1))
    x = point_at(5)
    dofs = dolfinx.fem.locate_dofs_geometrical(V, x)
    nodal_value = 42
    bc = dolfinx.fem.dirichletbc(ScalarType(nodal_value), dofs, V)
    ndofs = bc.dof_indices()[1]
    assert ndofs == 1
    assert bc.g.value == nodal_value


def test_function_space() -> None:
    n = 101
    domain = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, n, n, dolfinx.mesh.CellType.quadrilateral
    )
    V = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 2))

    h = 1.0 / n
    my_point = point_at(np.array([h * 2, h * 5]))

    dofs = dolfinx.fem.locate_dofs_geometrical(V, my_point)
    bc = dolfinx.fem.dirichletbc(ScalarType(42), dofs, V)
    ndofs = bc.dof_indices()[1]
    assert ndofs == 1
    assert bc.g.value == 42


def test_vector_function_space() -> None:
    n = 101
    domain = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, n, n, dolfinx.mesh.CellType.quadrilateral
    )
    V = dolfinx.fem.VectorFunctionSpace(domain, ("Lagrange", 2))

    # note the inconsistency in specifying the coordinates
    # this is handled by `to_floats`
    points = [0, [1.0], [0.0, 1.0], [1.0, 1.0, 0.0]]
    nodal_dofs = np.array([], dtype=np.int32)
    for x in points:
        dofs = dolfinx.fem.locate_dofs_geometrical(V, point_at(x))
        bc = dolfinx.fem.dirichletbc(np.array([0, 0], dtype=ScalarType), dofs, V)
        nodal_dofs = np.append(nodal_dofs, bc.dof_indices()[0])
    assert nodal_dofs.size == 8


if __name__ == "__main__":
    test_function_space()
    test_vector_function_space()
    test_type_error()
