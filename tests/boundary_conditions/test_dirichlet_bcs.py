"""Based on Philipp Diercks implementation for multi"""

import dolfinx
import numpy as np
import pytest
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from fenicsxconcrete.boundary_conditions.bcs import BoundaryConditions
from fenicsxconcrete.boundary_conditions.boundary import create_facet_tags, plane_at

"""Note: topological vs. geometrical

It seems that `locate_dofs_geometrical` does not work with V.sub
since at some point the dof coordinates need to be tabulated
which is not possible for a subspace.
However, one could always first locate the entities geometrically
if this is more convenient.

```python
from dolfinx.fem import dirichletbc
from dolfinx.mesh import locate_entities_boundary, locate_dofs_topological

def plane_at(coordinate, dim):

    def boundary(x):
        return np.isclose(x[dim], coordinate)

    return boundary

bottom = plane_at(0., 1)

bottom_boundary_facets = locate_entities_boundary(
    domain, domain.topology.dim - 1, bottom
)
bottom_boundary_dofs_y = locate_dofs_topological(
    V.sub(1), domain.topology.dim - 1, bottom_boundary_facets
)
fix_uy = dirichletbc(ScalarType(0), bottom_boundary_dofs_y, V.sub(1))
```

"""


def test_vector_geom() -> None:
    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, dolfinx.mesh.CellType.quadrilateral)
    V = dolfinx.fem.VectorFunctionSpace(domain, ("Lagrange", 2))

    bc_handler = BoundaryConditions(domain, V)

    def left(x):
        return np.isclose(x[0], 0.0)

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    # entire boundary; should have 64 * 2 dofs
    # constrain entire boundary only for the x-component
    boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
    bc_handler.add_dirichlet_bc(ScalarType(0), boundary_facets, sub=0, method="topological", entity_dim=fdim)
    # constrain left boundary as well
    zero = np.array([0.0, 0.0], dtype=ScalarType)
    bc_handler.add_dirichlet_bc(zero, left, method="geometrical")

    # use a Constant and constrain same dofs again for fun
    bc_handler.add_dirichlet_bc(
        dolfinx.fem.Constant(domain, ScalarType(0.0)),
        boundary_facets,
        sub=0,
        entity_dim=fdim,
    )

    bcs = bc_handler.bcs
    ndofs = 0
    for bc in bcs:
        ndofs += bc.dof_indices()[1]

    assert ndofs == 64 + 34 + 64


def test_vector_geom_component_wise() -> None:
    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, dolfinx.mesh.CellType.quadrilateral)
    V = dolfinx.fem.VectorFunctionSpace(domain, ("Lagrange", 2))

    bc_handler = BoundaryConditions(domain, V)

    def left(x):
        return np.isclose(x[0], 0.0)

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    zero = ScalarType(0.0)
    bc_handler.add_dirichlet_bc(zero, left, method="geometrical", sub=0, entity_dim=fdim)

    bcs = bc_handler.bcs
    ndofs = 0
    for bc in bcs:
        ndofs += bc.dof_indices()[1]

    assert ndofs == 17


def test_scalar_geom() -> None:
    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    V = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 2))

    bc_handler = BoundaryConditions(domain, V)

    def left(x):
        return np.isclose(x[0], 0.0)

    bc_handler.add_dirichlet_bc(ScalarType(0), left, method="geometrical")

    bcs = bc_handler.bcs
    my_bc = bcs[0]

    ndofs = my_bc.dof_indices()[1]
    all_ndofs = domain.comm.allreduce(ndofs, op=MPI.SUM)
    assert all_ndofs == 17
    assert my_bc.g.value == 0.0


def test_scalar_topo() -> None:
    n = 20
    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    V = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 2))

    bc_handler = BoundaryConditions(domain, V)

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    # entire boundary; should have (n+1+n)*4 - 4 = 8n dofs
    boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
    bc_handler.add_dirichlet_bc(ScalarType(0), boundary_facets, entity_dim=fdim)

    bcs = bc_handler.bcs
    my_bc = bcs[0]

    ndofs = my_bc.dof_indices()[1]
    all_ndofs = domain.comm.allreduce(ndofs, op=MPI.SUM)
    assert all_ndofs == 8 * n
    assert my_bc.g.value == 0.0


def test_dirichletbc() -> None:
    """add instance of dolfinx.fem.dirichletbc"""
    n = 20
    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    V = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 2))
    bc_handler = BoundaryConditions(domain, V)
    dofs = dolfinx.fem.locate_dofs_geometrical(V, plane_at(0.0, "x"))
    bc = dolfinx.fem.dirichletbc(ScalarType(0), dofs, V)
    assert not bc_handler.has_dirichlet
    bc_handler.add_dirichlet_bc(bc)
    assert bc_handler.has_dirichlet


def test_runtimeerror_geometrical() -> None:
    """test method geometrical raises RuntimeError if sub
    is not None"""
    n = 20
    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    V = dolfinx.fem.VectorFunctionSpace(domain, ("Lagrange", 2))
    Vsub = V.sub(0)
    bottom = plane_at(0.0, "y")
    with pytest.raises(RuntimeError):
        dolfinx.fem.locate_dofs_geometrical(Vsub, bottom)


def test_boundary_as_int() -> None:
    n = 5
    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    V = dolfinx.fem.VectorFunctionSpace(domain, ("Lagrange", 2))
    marker = 1011
    bottom = {"bottom": (marker, plane_at(0.0, "y"))}
    ft, marked = create_facet_tags(domain, bottom)

    bch_wo_ft = BoundaryConditions(domain, V)
    bc_handler = BoundaryConditions(domain, V, facet_tags=ft)

    zero = ScalarType(0.0)
    with pytest.raises(AttributeError):
        bch_wo_ft.add_dirichlet_bc(zero, boundary=0, sub=0, entity_dim=1)
    with pytest.raises(ValueError):
        bc_handler.add_dirichlet_bc(zero, boundary=0, sub=0, entity_dim=1)
    assert not bc_handler.has_dirichlet
    bc_handler.add_dirichlet_bc(zero, boundary=marker, sub=0, entity_dim=1)
    assert bc_handler.has_dirichlet


def test_value_interpolation() -> None:
    n = 50
    domain = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, n)
    V = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 2))
    my_value = 17.2

    def expression(x):
        return np.ones_like(x[0]) * my_value

    def everywhere(x):
        return np.full(x[0].shape, True, dtype=bool)

    bc_handler = BoundaryConditions(domain, V)
    bc_handler.add_dirichlet_bc(expression, everywhere, entity_dim=0)
    bc = bc_handler.bcs[0]
    dofs = bc.dof_indices()[0]
    assert np.allclose(np.ones_like(dofs) * my_value, bc.g.x.array[dofs])


def test_clear() -> None:
    n = 2
    domain = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, n)
    V = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 2))
    bc_handler = BoundaryConditions(domain, V)
    dofs = dolfinx.fem.locate_dofs_geometrical(V, plane_at(0.0, "x"))
    bc = dolfinx.fem.dirichletbc(ScalarType(0), dofs, V)
    assert not bc_handler.has_dirichlet
    bc_handler.add_dirichlet_bc(bc)
    assert bc_handler.has_dirichlet
    bc_handler.clear(neumann=False)
    assert not bc_handler.has_dirichlet


if __name__ == "__main__":
    test_scalar_geom()
    test_scalar_topo()
    test_vector_geom()
    test_vector_geom_component_wise()
    test_dirichletbc()
    test_runtimeerror_geometrical()
    test_boundary_as_int()
    test_value_interpolation()
    test_clear()
