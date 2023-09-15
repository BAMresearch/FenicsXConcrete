from collections.abc import Callable

import dolfinx as df
import numpy as np
import pint
import ufl
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from fenicsxconcrete.boundary_conditions.bcs import BoundaryConditions
from fenicsxconcrete.boundary_conditions.boundary import plane_at
from fenicsxconcrete.experimental_setup.base_experiment import Experiment
from fenicsxconcrete.util import Parameters, QuadratureRule, ureg


class AmMultipleLayers(Experiment):
    """sets up a simple layered structure

    all layers of the same height are on top of each other, the boundary on the bottom is fixed
    the mesh includes all (activation via pseudo-density)

    Attributes:
        parameters : parameter dictionary with units
        p : parameter dictionary without units

    """

    def __init__(self, parameters: dict[str, pint.Quantity]):
        """initializes the object, for the rest, see base class

        Args:
            parameters: dictionary containing the required parameters for the experiment set-up
                        see default_parameters for a first guess

        """

        super().__init__(parameters)

    @staticmethod
    def default_parameters() -> dict[str, pint.Quantity]:
        """sets up a working set of parameter values as example

        Returns:
            dictionary with a working set of the required parameter

        """

        setup_parameters = {}
        setup_parameters["degree"] = 2 * ureg("")  # polynomial degree
        # geometry
        setup_parameters["dim"] = 2 * ureg("")
        setup_parameters["num_layers"] = 10 * ureg("")  # number of layers in y
        setup_parameters["layer_length"] = 0.5 * ureg("m")  # x_dimension
        setup_parameters["layer_height"] = 0.01 * ureg("m")  # Dy dimension
        # only relevant for 3D case [z-dimension]
        setup_parameters["layer_width"] = 0.05 * ureg("m")

        # mesh
        setup_parameters["num_elements_layer_length"] = 10 * ureg("")
        setup_parameters["num_elements_layer_height"] = 1 * ureg("")
        # only relevant for 3D case
        setup_parameters["num_elements_layer_width"] = 2 * ureg("")

        return setup_parameters

    def setup(self) -> None:
        """defines the mesh for 2D and 3D

        Raises:
            ValueError: if dimension (self.p["dim"]) is not 2 or 3
        """

        self.logger.debug("setup mesh for %s", self.p["dim"])

        if self.p["dim"] == 2:
            self.mesh = df.mesh.create_rectangle(
                comm=MPI.COMM_WORLD,
                points=[(0.0, 0.0), (self.p["layer_length"], self.p["num_layers"] * self.p["layer_height"])],
                n=(self.p["num_elements_layer_length"], self.p["num_layers"] * self.p["num_elements_layer_height"]),
                cell_type=df.mesh.CellType.quadrilateral,
            )
        elif self.p["dim"] == 3:
            self.mesh = df.mesh.create_box(
                comm=MPI.COMM_WORLD,
                points=[
                    (0.0, 0.0, 0.0),
                    (self.p["layer_length"], self.p["layer_width"], self.p["num_layers"] * self.p["layer_height"]),
                ],
                n=[
                    self.p["num_elements_layer_length"],
                    self.p["num_elements_layer_width"],
                    self.p["num_layers"] * self.p["num_elements_layer_height"],
                ],
                cell_type=df.mesh.CellType.hexahedron,
            )
        else:
            raise ValueError(f'wrong dimension: {self.p["dim"]} is not implemented for problem setup')

    def create_displacement_boundary(self, V: df.fem.FunctionSpace) -> list[df.fem.bcs.DirichletBCMetaClass]:
        """defines displacement boundary as fixed at bottom

        Args:
            V: function space

        Returns:
            list of dirichlet boundary conditions

        """

        bc_generator = BoundaryConditions(self.mesh, V)

        if self.p["dim"] == 2:
            # fix dofs at bottom
            bc_generator.add_dirichlet_bc(
                np.array([0.0, 0.0], dtype=ScalarType),
                boundary=self.boundary_bottom(),
                method="geometrical",
                entity_dim=self.mesh.topology.dim - 1,  # line
            )

        elif self.p["dim"] == 3:
            # fix dofs at bottom
            bc_generator.add_dirichlet_bc(
                np.array([0.0, 0.0, 0.0], dtype=ScalarType),
                boundary=self.boundary_bottom(),
                method="geometrical",
                entity_dim=self.mesh.topology.dim - 1,  # surface
            )

        return bc_generator.bcs

    def create_body_force_am(
        self, v: ufl.argument.Argument, q_fd: df.fem.Function, rule: QuadratureRule
    ) -> ufl.form.Form:
        """defines body force for am experiments

        element activation via pseudo density and incremental loading via parameter ["load_time"] computed in class concrete_am

        Args:
            v: test function
            q_fd: quadrature function given the loading increment where elements are active
            rule: rule for the quadrature function

        Returns:
            form for body force

        """

        force_vector = np.zeros(self.p["dim"])
        force_vector[-1] = -self.p["rho"] * self.p["g"]  # works for 2D and 3D

        f = df.fem.Constant(self.mesh, ScalarType(force_vector))
        L = q_fd * ufl.dot(f, v) * rule.dx

        return L
