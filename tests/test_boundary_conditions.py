import numpy as np
import dolfinx as df

from mpi4py import MPI
from fenicsxconcrete.experimental_setup.boundary_conditions import create_clamped_boundary

import pytest

def create_simple_geometry(dimension):

    if dimension == 1:
        msh = df.mesh.create_unit_interval(comm = MPI.COMM_WORLD, nx = 16)

    if dimension == 2:
        msh = df.mesh.create_unit_square(comm=MPI.COMM_WORLD, nx= 16, ny=16)
    
    if dimension == 3:
        msh = df.mesh.create_unit_cube(comm= MPI.COMM_WORLD, nx = 16, ny = 16, nz = 16)
    
    return msh

@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("coord", [0, 1, 2])
def test_clamped_boundary_condition(dim, coord):

    msh = create_simple_geometry(dim)
    V = df.fem.FunctionSpace(msh, ("Lagrange", 1))

    bc = create_clamped_boundary(msh, V, 0.0, coord)
    print("STOP")
