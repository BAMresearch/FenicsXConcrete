"""Easy definition of Dirichlet and Neumann BCs."""

from collections.abc import Callable

import dolfinx
import numpy as np
import ufl
from dolfinx.fem.bcs import DirichletBCMetaClass


def get_boundary_dofs(V: dolfinx.fem.FunctionSpace, marker: Callable) -> np.ndarray:
    """Returns dofs on the boundary specified by geometrical `marker`."""
    domain = V.mesh
    tdim = domain.topology.dim
    fdim = tdim - 1
    entities = dolfinx.mesh.locate_entities_boundary(domain, fdim, marker)
    dofs = dolfinx.fem.locate_dofs_topological(V, fdim, entities)
    g = dolfinx.fem.Function(V)
    bc = dolfinx.fem.dirichletbc(g, dofs)
    dof_indices = bc.dof_indices()[0]
    return dof_indices


# adapted version of MechanicsBCs by Thomas Titscher
class BoundaryConditions:
    """Handles Dirichlet and Neumann boundary conditions.

    Attributes:
        domain: The computational domain.
        V: The finite element space.
    """

    def __init__(
        self,
        domain: dolfinx.mesh.Mesh,
        space: dolfinx.fem.FunctionSpace,
        facet_tags: np.ndarray | None = None,
    ) -> None:
        """Initializes the instance based on domain and FE space.

        It sets up lists to hold the Dirichlet and Neumann BCs
        as well as the required `ufl` objects to define Neumann
        BCs if `facet_tags` is not None.

        Args:
          domain: The computational domain.
          space: The finite element space.
          facet_tags: The mesh tags defining boundaries.
        """

        self.domain = domain
        self.V = space

        # create connectivity
        tdim = domain.topology.dim
        fdim = tdim - 1
        domain.topology.create_connectivity(fdim, tdim)

        # list of dirichlet boundary conditions
        self._bcs = []

        # handle facets and measure for neumann bcs
        self._neumann_bcs = []
        self._facet_tags = facet_tags
        self._ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
        self._v = ufl.TestFunction(space)

    def add_dirichlet_bc(
        self,
        value: (
            dolfinx.fem.Function
            | dolfinx.fem.Constant
            | dolfinx.fem.DirichletBCMetaClass
            | np.ndarray
            | Callable
        ),
        boundary: int | np.ndarray | Callable | None = None,
        sub: int = None,
        method: str = "topological",
        entity_dim: int | None = None,
    ) -> None:
        """Adds a Dirichlet bc.

        Args:
          value: Anything that *might* be used to define the Dirichlet function.
            It can be a `Function`, a `Callable` which is then interpolated
            or an already existing Dirichlet BC, or ... (see type hint).
          boundary: The part of the boundary whose dofs should be constrained.
            This can be a callable defining the boundary geometrically or
            an array of entity tags or an integer marking the boundary if
            `facet_tags` is not None.
          sub: If `sub` is not None the subspace `V.sub(sub)` will be
            constrained.
          method: A hint which method should be used to locate the dofs.
            Choices: 'topological' or 'geometrical'.
          entity_dim: The dimension of the entities to be located
            topologically. Note that `entity_dim` is required if `sub`
            is not None and `method=geometrical`.
        """
        if isinstance(value, dolfinx.fem.DirichletBCMetaClass):
            self._bcs.append(value)
        else:
            assert method in ("topological", "geometrical")
            V = self.V.sub(sub) if sub is not None else self.V

            # if sub is not None and method=="geometrical"
            # dolfinx.fem.locate_dofs_geometrical(V, boundary) will raise a RuntimeError
            # because dofs of a subspace cannot be tabulated
            topological = method == "topological" or sub is not None

            if topological:
                assert entity_dim is not None

                if isinstance(boundary, int):
                    try:
                        facets = self._facet_tags.find(boundary)
                    except AttributeError:
                        raise AttributeError("There are no facet tags defined!")
                    if facets.size < 1:
                        raise ValueError(
                            f"Not able to find facets tagged with value {boundary=}."
                        )
                elif isinstance(boundary, np.ndarray):
                    facets = boundary
                else:
                    facets = dolfinx.mesh.locate_entities_boundary(
                        self.domain, entity_dim, boundary
                    )

                dofs = dolfinx.fem.locate_dofs_topological(V, entity_dim, facets)
            else:
                dofs = dolfinx.fem.locate_dofs_geometrical(V, boundary)

            try:
                bc = dolfinx.fem.dirichletbc(value, dofs, V)
            except TypeError:
                # value is Function and V cannot be passed
                # TODO understand 4th constructor
                # see dolfinx/fem/bcs.py line 127
                bc = dolfinx.fem.dirichletbc(value, dofs)
            except AttributeError:
                # value has no Attribute `dtype`
                f = dolfinx.fem.Function(V)
                f.interpolate(value)
                bc = dolfinx.fem.dirichletbc(f, dofs)

            self._bcs.append(bc)

    def add_neumann_bc(self, marker: int, value: dolfinx.fem.Constant) -> None:
        """Adds a Neumann BC.

        Args:
          marker: The id of the boundary where Neumann BC should be applied.
          value: The Neumann data, e.g. a traction vector. This has
            to be a valid `ufl` object.
        """
        if marker not in self._facet_tags.values:
            raise ValueError(f"No facet tags defined for {marker=}.")

        self._neumann_bcs.append([value, marker])

    @property
    def has_neumann(self) -> bool:
        return len(self._neumann_bcs) > 0

    @property
    def has_dirichlet(self) -> bool:
        return len(self._bcs) > 0

    @property
    def bcs(self) -> list[dolfinx.fem.DirichletBCMetaClass]:
        """The list of Dirichlet BCs."""
        return self._bcs

    def clear(self, dirichlet: bool = True, neumann: bool = True) -> None:
        """Clears list of Dirichlet and/or Neumann BCs."""
        if dirichlet:
            self._bcs.clear()
        if neumann:
            self._neumann_bcs.clear()

    @property
    def neumann_bcs(self) -> ufl.form.Form:
        """The ufl ufl.form.Form of (sum of) Neumann BCs"""
        r = 0
        for expression, marker in self._neumann_bcs:
            r += ufl.inner(expression, self._v) * self._ds(marker)
        return r
