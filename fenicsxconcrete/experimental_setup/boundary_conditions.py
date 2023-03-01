"""
Boundary conditions to be used in ExperimentalSetup definition.
Created: 01.03.2023
Last version: 01.03.2023
"""

import numpy as np
from dolfinx import mesh, fem
from dolfinx.fem import FunctionSpace
from dolfinx.mesh import Mesh

from petsc4py.PETSc import ScalarType

##################################
# BOUNDARY CONDITIONS CONSTRUCTORS


def create_clamped_boundary(
    domain: Mesh, v_function_space: FunctionSpace, side_coord_value: float, coord: int
):
    """
    Clamped boundary condition on a mesh.

    Args:
        domain: Domain on which the clamped boundary is to be applied
        V: Function space where the BC is to be defined
        side_coords_value: coordinate values close to which the BC must be applied
        coord: coordinate index where the BC is to be applied
    """

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, boundary_side(side_coord_value, coord)
    )

    u_d = np.array([0] * (fdim + 1), dtype=ScalarType)
    bc_output = fem.dirichletbc(
        u_d,
        fem.locate_dofs_topological(v_function_space, fdim, boundary_facets),
        v_function_space,
    )
    return bc_output


#######################
# GEOMETRY DESCRIPTIONS


def boundary_side(side_coord, coord):
    """
    Returns the side of the boundary

    Args:
        side_coord: value close to which it is conider as boundary
        coord: index for the coordinates
    """
    return lambda x: np.isclose(x[coord], side_coord)


def boundary_full(boundary: np.ndarray):
    """
    Returns the full boundary

    Args:
        boundary: boundary of the mesh
    """
    return boundary


def boundary_empty():
    """
    Returns an empty boundary
    """
    return None
