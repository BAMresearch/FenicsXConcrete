"""Easy definition of boundaries."""
import typing
from collections.abc import Callable

import dolfinx
import numpy as np

"""Design

old dolfin:
    here on needed a SubDomain object that defined the boundary geometrically.
    SubDomain could then be passed to DirichletBC.
    Therefore, fenics_helpers.boundary was used to conveniently define
    boundaries geometrically (would return a SubDomain).

dolfinx:
    input to dirichletbc is now:
        1. (Function, array)
        2. ([Constant, array], array, FunctionSpace)
    The array are the boundary_dofs which are determined via
    `locate_dofs_topological` or `locate_dofs_geometrical`.

    Thus, multi.boundary could provide functions to:
        (a) define callables that define complex geometry as input to
            locate_dofs_geometrical.
        (b) define functions that compute entities of the mesh and pass
            this array to locate_dofs_topological.

    (b) might use dolfinx.mesh.locate_entities and
        dolfinx.mesh.locate_entities_boundary

    Args:
        mesh: dolfinx.mesh.Mesh
        dim: tdim of the entities
        marker: function that takes an array of points x and
                returns an array of booleans

    --> therefore, use of locate_dofs_topological again boils down
        to a geometrical description of the boundary to be defined.
        The only difference is the possibility to filter wrt the tdim.
        (this is not possible with locate_dofs_geometrical)

"""


def plane_at(coordinate: float, dim: str | int) -> Callable:
    """Defines a plane where `x[dim]` equals `coordinate`."""

    if dim in ["x", "X"]:
        dim = 0
    if dim in ["y", "Y"]:
        dim = 1
    if dim in ["z", "Z"]:
        dim = 2

    assert dim in (0, 1, 2)

    def boundary(x):
        return np.isclose(x[dim], coordinate)

    return boundary


def line_at(coordinates: list[float], dims: list[str | int]) -> Callable:
    """return callable that determines boundary geometrically

    Parameters
    ----------
    coordinates
    dims
    """
    assert len(coordinates) == 2
    assert len(dims) == 2

    # transform x,y,z str into integer
    for i, dim in enumerate(dims):
        if dim in ["x", "X"]:
            dims[i] = 0
        elif dim in ["y", "Y"]:
            dims[i] = 1
        elif dim in ["z", "Z"]:
            dims[i] = 2
        assert dims[i] in (0, 1, 2)

    assert dims[0] != dims[1]

    def boundary(x):
        return np.logical_and(
            np.isclose(x[dims[0]], coordinates[0]),
            np.isclose(x[dims[1]], coordinates[1]),
        )

    return boundary


def within_range(
    start: typing.Iterable[int] | typing.Iterable[float],
    end: typing.Iterable[int] | typing.Iterable[float],
    tol: float = 1e-6,
) -> Callable:
    """Defines a range.

    It is best used together with `dolfinx.mesh.locate_entities_boundary`
    and topological definition of the Dirichlet BC, because the Callable
    will mark the whole range and not just the boundary.

    Args:
      start: The start point of the range.
      end: The end point of the range.
    """
    start = to_floats(start)
    end = to_floats(end)

    # adjust the values such that start < end for all dimensions
    assert len(start) == 3
    assert len(start) == len(end)
    for i in range(len(start)):
        if start[i] > end[i]:
            start[i], end[i] = end[i], start[i]

    def boundary(x):
        def in_range(i):
            return np.logical_and(x[i] >= start[i] - tol, x[i] <= end[i] + tol)

        xy = np.logical_and(in_range(0), in_range(1))
        return np.logical_and(xy, in_range(2))

    return boundary


def point_at(coord: typing.Iterable[int] | typing.Iterable[float]) -> Callable:
    """Defines a point."""
    p = to_floats(coord)

    def boundary(x):
        return np.logical_and(
            np.logical_and(np.isclose(x[0], p[0]), np.isclose(x[1], p[1])),
            np.isclose(x[2], p[2]),
        )

    return boundary


def show_marked(
    domain: dolfinx.mesh.Mesh,
    marker: Callable,
    filename: str | None = None,
) -> None:
    """Shows dof coordinates marked by `marker`.

    Notes:
      This is useful for debugging boundary conditions.
      Currently this only works for domains of topological
      dimension 2.

    Args:
      domain: The computational domain.
      marker: A function that takes an array of points ``x`` with shape
        ``(gdim, num_points)`` and returns an array of booleans of
        length ``num_points``, evaluating to ``True`` for entities whose
        degree-of-freedom should be returned.
      filename: Save figure to this path.
        If None, the figure is shown (default).
    """
    import matplotlib.pyplot as plt

    tdim = domain.topology.dim
    if tdim in (1, 3):
        raise NotImplementedError(
            f"Not implemented for mesh of topological dimension {tdim=}."
        )

    V = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 1))
    dofs = dolfinx.fem.locate_dofs_geometrical(V, marker)
    u = dolfinx.fem.Function(V)
    bc = dolfinx.fem.dirichletbc(u, dofs)
    x_dofs = V.tabulate_dof_coordinates()
    x_dofs = x_dofs[:, :2]
    marked = x_dofs[bc.dof_indices()[0]]

    plt.figure(1)
    x, y = x_dofs.T
    plt.scatter(x, y, facecolors="none", edgecolors="k", marker="o")
    xx, yy = marked.T
    plt.scatter(xx, yy, facecolors="r", edgecolors="none", marker="o")

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()  # pragma: no cover


def to_floats(x: typing.Iterable[int] | typing.Iterable[float]) -> list[float]:
    """Converts `x` to a 3d coordinate."""
    floats = []
    try:
        for v in x:
            floats.append(float(v))
        while len(floats) < 3:
            floats.append(0.0)
    except TypeError:
        floats = [float(x), 0.0, 0.0]

    return floats


def create_facet_tags(
    mesh: dolfinx.mesh.Mesh, boundaries: dict[str, tuple[int, Callable]]
) -> tuple[np.ndarray, dict[str, int]]:
    """Creates facet tags for the given mesh and boundaries.

    This code is part of the FEniCSx tutorial
    by Jørgen S. Dokken.
    See https://jsdokken.com/dolfinx-tutorial/chapter3/robin_neumann_dirichlet.html?highlight=sorted_facets#implementation # noqa: E501

    Args:
      mesh: The computational domain.
      boundaries: The definition of boundaries where each key is a string
        and each value is a tuple of an integer and a marker function.

    Returns:
      A tuple (facet_tags, marked_boundary) where facet_tags is an array
      with dtype int and marked_boundary is a dict where each key is a string
      and each value is an int.
    """

    facet_indices, facet_markers = [], []
    fdim = mesh.topology.dim - 1
    marked_boundary = {}
    for key, (marker, locator) in boundaries.items():
        facets = dolfinx.mesh.locate_entities(mesh, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
        if facets.size > 0:
            marked_boundary[key] = marker
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tags = dolfinx.mesh.meshtags(
        mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets]
    )
    return facet_tags, marked_boundary
