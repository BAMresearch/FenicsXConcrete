from collections.abc import Callable

import dolfinx as df
import numpy as np
import pint
import ufl
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from fenicsxconcrete.boundary_conditions.bcs import BoundaryConditions
from fenicsxconcrete.boundary_conditions.boundary import line_at, plane_at, point_at
from fenicsxconcrete.experimental_setup.base_experiment import Experiment
from fenicsxconcrete.helper import Parameters
from fenicsxconcrete.unit_registry import ureg


class SimpleBeam(Experiment):
    """Sets up a simply supported beam, fix on the left

    Attributes:
        parameters : parameter dictionary with units
        p : parameter dictionary without units

    """

    def __init__(self, parameters: dict[str, pint.Quantity]) -> None:
        """initializes the object,  for the rest, see base class

        Args:
            parameters: dictionary containing the required parameters for the experiment set-up
                        see default_parameters for a first guess

        """

        # initialize default parameters for the setup
        default_p = Parameters()
        default_p["degree"] = 2 * ureg("")  # polynomial degree

        # updating parameters, overriding defaults
        default_p.update(parameters)

        super().__init__(default_p)

    def setup(self):
        """defines the mesh for 2D or 3D

        Raises:
            ValueError: if dimension (self.p["dim"]) is not 2 or 3
        """

        if self.p["dim"] == 2:
            self.mesh = df.mesh.create_rectangle(
                comm=MPI.COMM_WORLD,
                points=[(0.0, 0.0), (self.p["length"], self.p["height"])],
                n=(self.p["num_elements_length"], self.p["num_elements_height"]),
                cell_type=df.mesh.CellType.quadrilateral,
            )
        elif self.p["dim"] == 3:
            self.mesh = df.mesh.create_box(
                comm=MPI.COMM_WORLD,
                points=[
                    (0.0, 0.0, 0.0),
                    (self.p["length"], self.p["width"], self.p["height"]),
                ],
                n=[
                    self.p["num_elements_length"],
                    self.p["num_elements_width"],
                    self.p["num_elements_height"],
                ],
                cell_type=df.mesh.CellType.hexahedron,
            )
        else:
            raise ValueError(f'wrong dimension: {self.p["dim"]} is not implemented for problem setup')

    @staticmethod
    def default_parameters() -> dict[str, pint.Quantity]:
        """sets up a working set of parameter values as example

        Returns:
            dictionary with a working set of the required parameter

        """

        setup_parameters = {}
        setup_parameters["load"] = 10000 * ureg("N/m^2")
        setup_parameters["length"] = 1 * ureg("m")
        setup_parameters["height"] = 0.3 * ureg("m")
        setup_parameters["width"] = 0.3 * ureg("m")  # only relevant for 3D case
        setup_parameters["dim"] = 3 * ureg("")
        setup_parameters["num_elements_length"] = 10 * ureg("")
        setup_parameters["num_elements_height"] = 3 * ureg("")
        # only relevant for 3D case
        setup_parameters["num_elements_width"] = 3 * ureg("")

        return setup_parameters

    def create_displacement_boundary(self, V) -> list:
        """defines displacement boundary as fixed at bottom

        Args:
            V: function space

        Returns:
            list of dirichlet boundary conditions

        """

        bc_generator = BoundaryConditions(self.mesh, V)

        if self.p["dim"] == 2:
            # fix line in the left
            bc_generator.add_dirichlet_bc(
                np.array([0.0, 0.0], dtype=ScalarType),
                boundary=self.boundary_left(),
                method="geometrical",
            )
            # line with dof in x direction on the right
            bc_generator.add_dirichlet_bc(np.float64(0.0), self.boundary_right(), 1, "geometrical", 0)

        elif self.p["dim"] == 3:
            # fix line in the left
            bc_generator.add_dirichlet_bc(
                np.array([0.0, 0.0, 0.0], dtype=ScalarType),
                boundary=self.boundary_left(),
                method="geometrical",
            )
            # line with dof in x direction on the right
            bc_generator.add_dirichlet_bc(np.float64(0.0), self.boundary_right(), 1, "geometrical", 0)
            bc_generator.add_dirichlet_bc(np.float64(0.0), self.boundary_right(), 2, "geometrical", 0)

        return bc_generator.bcs

    def create_body_force(self, v: ufl.argument.Argument) -> ufl.form.Form:
        """defines body force

        Args:
            v: test function

        Returns:
            form for body force

        """

        force_vector = np.zeros(self.p["dim"])
        force_vector[-1] = -self.p["rho"] * self.p["g"]  # works for 2D and 3D

        f = df.fem.Constant(self.mesh, ScalarType(force_vector))
        L = ufl.dot(f, v) * ufl.dx

        return L

    def boundary_left(self) -> Callable:
        """specifies boundary at bottom

        Returns:
            fct defining boundary

        """

        if self.p["dim"] == 2:
            return point_at([0, 0])
        elif self.p["dim"] == 3:
            return line_at([0, 0], ["x", "z"])

    def boundary_right(self) -> Callable:
        """specifies boundary at bottom

        Returns:
            fct defining boundary

        """

        if self.p["dim"] == 2:
            return point_at([self.p["length"], 0])
        elif self.p["dim"] == 3:
            return line_at([self.p["length"], 0], ["x", "z"])

    def create_force_boundary(self, v: ufl.argument.Argument) -> ufl.form.Form:
        """distributed load on top of beam

        Args:
            v: test function

        Returns:
            form for force boundary

        """

        # TODO: make this more pretty!!!
        #       can we use Philipps boundary classes here?

        facet_indices, facet_markers = [], []
        fdim = self.mesh.topology.dim - 1

        locator = lambda x: np.isclose(x[fdim], self.p["height"])
        facets = df.mesh.locate_entities(self.mesh, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, 1))
        facet_indices = np.hstack(facet_indices).astype(np.int32)
        facet_markers = np.hstack(facet_markers).astype(np.int32)
        sorted_facets = np.argsort(facet_indices)
        facet_tag = df.mesh.meshtags(self.mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

        _ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=facet_tag)

        force_vector = np.zeros(self.p["dim"])
        force_vector[-1] = -self.p["load"]
        T = df.fem.Constant(self.mesh, ScalarType(force_vector))
        L = ufl.dot(T, v) * _ds(1)

        return L
