"""Based on Philipp Diercks implementation for multi"""

import dolfinx
import ufl
import numpy as np
from petsc4py import PETSc


def get_boundary_dofs(V, marker):
    """get dofs on the boundary specified by `marker`

    Parameters
    ----------
    V : dolfinx.fem.FunctionSpace
        The FE space.
    marker : callable
        A callable defining the boundary.
    """
    domain = V.mesh
    gdim = domain.geometry.dim
    tdim = domain.topology.dim
    fdim = tdim - 1
    entities = dolfinx.mesh.locate_entities_boundary(domain, fdim, marker)
    dofs = dolfinx.fem.locate_dofs_topological(V, fdim, entities)
    bc = dolfinx.fem.dirichletbc(np.array((0,) * gdim, dtype=PETSc.ScalarType), dofs, V)
    dof_indices = bc.dof_indices()[0]
    return dof_indices


# adapted version of MechanicsBCs by Thomas Titscher
class BoundaryConditions:
    """Handles dirichlet and neumann boundary conditions

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        Computational domain of the problem.
    space : dolfinx.fem.FunctionSpace
        Finite element space defined on the domain.
    facet_tags : optional, dolfinx.mesh.MeshTags
        The mesh tags defining boundaries.

    """

    def __init__(self, domain, space, facet_tags=None):
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
        self, value, boundary=None, sub=None, method="topological", entity_dim=None
    ):
        """add a Dirichlet bc

        Parameters
        ----------
        value : Function, Constant or np.ndarray or DirichletBCMetaClass
            The Dirichlet function or boundary condition.
        boundary : optional, callable or np.ndarray or int
            The part of the boundary whose dofs should be constrained.
            This can be a callable defining the boundary geometrically or
            an array of entity tags or an integer marking the boundary if
            `facet_tags` is not None.
        sub : optional, int
            If `sub` is not None the subspace `V.sub(sub)` will be constrained.
        method : optional, str
            A hint which method should be used to locate the dofs.
            Choice: 'topological' or 'geometrical'.
        entity_dim : optional, int
            The dimension of the entities to be located topologically.
            Note that `entity_dim` is required if `sub` is not None and
            `method=geometrical`.
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
                        raise ValueError(f"Not able to find facets tagged with value {boundary=}.")
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

    def add_neumann_bc(self, marker, value):
        """adds a Neumann BC.

        Parameters
        ----------
        marker : int
        value : some ufl type
            The neumann data, e.g. traction vector.

        """
        if isinstance(marker, int):
            assert marker in self._facet_tags.values

        self._neumann_bcs.append([value, marker])

    @property
    def has_neumann(self):
        return len(self._neumann_bcs) > 0

    @property
    def has_dirichlet(self):
        return len(self._bcs) > 0

    @property
    def bcs(self):
        """returns list of dirichlet bcs"""
        return self._bcs

    def clear(self, dirichlet=True, neumann=True):
        """clear list of boundary conditions"""
        if dirichlet:
            self._bcs.clear()
        if neumann:
            self._neumann_bcs.clear()

    @property
    def neumann_bcs(self):
        """returns ufl form of (sum of) neumann bcs"""
        r = 0
        for expression, marker in self._neumann_bcs:
            r += ufl.inner(expression, self._v) * self._ds(marker)
        return r
