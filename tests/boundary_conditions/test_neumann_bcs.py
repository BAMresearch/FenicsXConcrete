import dolfinx
import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc

from fenicsxconcrete.boundary_conditions.bcs import BoundaryConditions
from fenicsxconcrete.boundary_conditions.boundary import create_facet_tags, plane_at


def test_constant_traction() -> None:
    n = 10
    domain = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, n, n, dolfinx.mesh.CellType.quadrilateral
    )
    V = dolfinx.fem.VectorFunctionSpace(domain, ("Lagrange", 1))
    rmarker = 12
    my_boundaries = {"right": (rmarker, plane_at(0.0, "x"))}
    ft, mb = create_facet_tags(domain, my_boundaries)
    bch = BoundaryConditions(domain, V, facet_tags=ft)

    tmax = 234.0
    traction = dolfinx.fem.Constant(domain, PETSc.ScalarType((tmax, 0.0)))
    assert not bch.has_neumann
    bch.add_neumann_bc(rmarker, traction)
    assert bch.has_neumann

    rhs = bch.neumann_bcs
    form = dolfinx.fem.form(rhs)
    vector = dolfinx.fem.petsc.create_vector(form)

    with vector.localForm() as v_loc:
        v_loc.set(0)
    dolfinx.fem.petsc.assemble_vector(vector, form)
    vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    f_ext = np.sum(vector[:])
    assert np.isclose(tmax, f_ext)

    bch.clear(dirichlet=False)
    assert not bch.has_neumann

    # try to add non-existent marker
    with pytest.raises(ValueError):
        bch.add_neumann_bc(666, traction)


if __name__ == "__main__":
    test_constant_traction()
