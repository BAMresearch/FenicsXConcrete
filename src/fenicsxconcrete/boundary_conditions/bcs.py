"""Based on Philipp Diercks implementation for multi"""

import logging

import dolfinx
import ufl
import numpy as np
from petsc4py import PETSc


def get_boundary_dofs(V, marker):
    """get dofs on the boundary"""
    domain = V.mesh
    gdim = domain.geometry.dim
    tdim = domain.topology.dim
    fdim = tdim - 1
    entities = dolfinx.mesh.locate_entities_boundary(domain, fdim, marker)
    dofs = dolfinx.fem.locate_dofs_topological(V, fdim, entities)
    bc = dolfinx.fem.dirichletbc(np.array((0,) * gdim, dtype=PETSc.ScalarType), dofs)
    dof_indices = bc.dof_indices()[0]
    return dof_indices

# adapted version of MechanicsBCs by Thomas Titscher
class BoundaryConditions:
    """Handles dirichlet and neumann boundary conditions

    Parameters
    ----------
    domain : Computational domain of the problem.
    space : Finite element space defined on the domain.
    facet_markers : The mesh tags defining boundaries.

    """

    def __init__(self, domain: dolfinx.mesh.Mesh, space:dolfinx.fem.FunctionSpace, facet_markers:dolfinx.mesh.MeshTagsMetaClass=None):

        self.logger = logging.getLogger("fenicsxconcrete.boundary_conditions.bcs.BoundaryConditions")
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
        self._facet_markers = facet_markers
        self._ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_markers)
        self._v = ufl.TestFunction(space)

    def add_dirichlet_bc(
        self, value, boundary=None, sub=None, method="topological", entity_dim=None
    ):
        """add a Dirichlet BC

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
            The entity dimension in case `method=topological`.
        """
        if boundary is None:
            assert isinstance(value, dolfinx.fem.DirichletBCMetaClass)
            self._bcs.append(value)
        else:
            assert method in ("topological", "geometrical")
            V = self.V.sub(sub) if sub is not None else self.V

            if method == "topological":
                assert entity_dim is not None

                if isinstance(boundary, int):
                    try:
                        facets = self._facet_tags.find(boundary)
                    except AttributeError as atterr:
                        raise atterr("There are no facet tags defined!")
                else:
                    facets = boundary

                dofs = dolfinx.fem.locate_dofs_topological(V, entity_dim, facets)
            else:
                if sub is not None:
                    assert entity_dim is not None
                    facets = dolfinx.mesh.locate_entities_boundary(
                        self.domain, entity_dim, boundary
                    )
                    dofs = dolfinx.fem.locate_dofs_topological(V, entity_dim, facets)
                else:
                    dofs = dolfinx.fem.locate_dofs_geometrical(V, boundary)

            if isinstance(value, (dolfinx.fem.Constant, np.ndarray, np.float64)):
                bc = dolfinx.fem.dirichletbc(value, dofs, V)
            else:
                try:
                    bc = dolfinx.fem.dirichletbc(value, dofs)
                except AttributeError:
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
            assert marker in self._facet_markers.values

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
