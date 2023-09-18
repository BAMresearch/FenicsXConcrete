from collections.abc import Callable

import dolfinx as df
import numpy as np
import pint
import ufl
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from fenicsxconcrete.boundary_conditions.bcs import BoundaryConditions
from fenicsxconcrete.experimental_setup.base_experiment import Experiment
from fenicsxconcrete.util import LogMixin, Parameters, ureg


class SimpleCube(Experiment):
    """sets up an uniaxial cube structure with displacement load

    2D unit square or 3D unit cube with uniaxial boundary conditions
    displacement controlled
    for material model testing

    Attributes:
        parameters: parameter dictionary with units
        p: parameter dictionary without units

    """

    def __init__(self, parameters: dict[str, pint.Quantity] | None = None) -> None:
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

        setup_parameters["height"] = 1 * ureg("m")
        setup_parameters["width"] = 1 * ureg("m")
        setup_parameters["length"] = 1 * ureg("m")
        setup_parameters["T_0"] = ureg.Quantity(20.0, ureg.degC)
        setup_parameters["T_bc"] = ureg.Quantity(20.0, ureg.degC)
        setup_parameters["dim"] = 3 * ureg("")
        setup_parameters["num_elements_length"] = 2 * ureg("")
        setup_parameters["num_elements_width"] = 2 * ureg("")
        setup_parameters["num_elements_height"] = 2 * ureg("")
        setup_parameters["strain_state"] = "uniaxial" * ureg("")

        return setup_parameters

    def setup(self) -> None:
        """Generates the mesh in 2D or 3D based on parameters

        Raises:
            ValueError: if dimension (self.p["dim"]) is not 2 or 3
        """

        self.logger.debug("setup mesh for %s", self.p["dim"])

        if self.p["dim"] == 2:
            # build a rectangular mesh
            self.mesh = df.mesh.create_rectangle(
                MPI.COMM_WORLD,
                [
                    [0.0, 0.0],
                    [self.p["length"], self.p["height"]],
                ],
                [self.p["num_elements_length"], self.p["num_elements_height"]],
                cell_type=df.mesh.CellType.quadrilateral,
            )
        elif self.p["dim"] == 3:
            self.mesh = df.mesh.create_box(
                MPI.COMM_WORLD,
                [
                    [0.0, 0.0, 0.0],
                    [self.p["length"], self.p["width"], self.p["height"]],
                ],
                [self.p["num_elements_length"], self.p["num_elements_width"], self.p["num_elements_height"]],
                cell_type=df.mesh.CellType.hexahedron,
            )

        else:
            raise ValueError(f"wrong dimension {self.p['dim']} for problem setup")

        # initialize variable top_displacement
        self.top_displacement = df.fem.Constant(domain=self.mesh, c=0.0)  # applied via fkt: apply_displ_load(...)
        self.use_body_force = False
        self.temperature_bc = df.fem.Constant(domain=self.mesh, c=self.p["T_bc"])

    def create_displacement_boundary(self, V: df.fem.FunctionSpace) -> list[df.fem.bcs.DirichletBCMetaClass]:
        """Defines the displacement boundary conditions

        Args:
            V :Function space of the structure

        Returns:
            list of DirichletBC objects, defining the boundary conditions
        """

        # define boundary conditions generator
        bc_generator = BoundaryConditions(self.mesh, V)

        if self.p["dim"] == 2:
            # uniaxial bcs
            bc_generator.add_dirichlet_bc(
                np.float64(0.0), boundary=self.boundary_bottom(), sub=1, method="geometrical", entity_dim=1
            )
            bc_generator.add_dirichlet_bc(
                np.float64(0.0), boundary=self.boundary_left(), sub=0, method="geometrical", entity_dim=1
            )

            if self.p["strain_state"] == "uniaxial":
                # displacement controlled
                bc_generator.add_dirichlet_bc(
                    self.top_displacement, boundary=self.boundary_top(), sub=1, method="geometrical", entity_dim=1
                )
            elif self.p["strain_state"] == "multiaxial":
                # displacement controlled
                bc_generator.add_dirichlet_bc(
                    self.top_displacement, boundary=self.boundary_top(), sub=1, method="geometrical", entity_dim=1
                )
                bc_generator.add_dirichlet_bc(
                    self.top_displacement, boundary=self.boundary_right(), sub=0, method="geometrical", entity_dim=1
                )
            else:
                raise ValueError(f'Strain_state value: {self.p["strain_state"]} is not implemented in 2D.')

        elif self.p["dim"] == 3:
            # uniaxial bcs
            bc_generator.add_dirichlet_bc(
                np.float64(0.0), boundary=self.boundary_bottom(), sub=2, method="geometrical", entity_dim=2
            )
            bc_generator.add_dirichlet_bc(
                np.float64(0.0), boundary=self.boundary_left(), sub=0, method="geometrical", entity_dim=2
            )
            bc_generator.add_dirichlet_bc(
                np.float64(0.0), boundary=self.boundary_front(), sub=1, method="geometrical", entity_dim=2
            )

            # displacement controlled
            if self.p["strain_state"] == "uniaxial":
                bc_generator.add_dirichlet_bc(
                    self.top_displacement, boundary=self.boundary_top(), sub=2, method="geometrical", entity_dim=2
                )
            elif self.p["strain_state"] == "multiaxial":
                bc_generator.add_dirichlet_bc(
                    self.top_displacement, boundary=self.boundary_top(), sub=2, method="geometrical", entity_dim=2
                )
                bc_generator.add_dirichlet_bc(
                    self.top_displacement, boundary=self.boundary_right(), sub=0, method="geometrical", entity_dim=2
                )
                bc_generator.add_dirichlet_bc(
                    self.top_displacement, boundary=self.boundary_back(), sub=1, method="geometrical", entity_dim=2
                )
            else:
                raise ValueError(f'Strain_state value: {self.p["strain_state"]} is not implemented in 3D.')

        return bc_generator.bcs

    def apply_displ_load(self, top_displacement: pint.Quantity | float) -> None:
        """Updates the applied displacement load

        Args:
            top_displacement: Displacement of the top boundary in mm, > 0 ; tension, < 0 ; compression

        """
        top_displacement.ito_base_units()
        self.top_displacement.value = top_displacement.magnitude

    def apply_temp_bc(self, T_bc: pint.Quantity | float) -> None:
        """Updates the applied temperature boundary condition

        Args:
            T_bc1: Temperature of the top boundary in degree Celsius

        """
        T_bc.ito_base_units()
        self.temperature_bc.value = T_bc.magnitude
        self.p["T_bc"] = T_bc.magnitude

    def apply_body_force(self) -> None:
        self.use_body_force = True

    def create_temperature_bcs(self, V: df.fem.FunctionSpace) -> list[df.fem.bcs.DirichletBCMetaClass]:
        """defines empty temperature boundary conditions (to be done in child)

        this function is abstract until there is a need for a material that does need a temperature boundary
        once that is required, just make this a normal function that returns an empty list

        Args:
            V: function space

        Returns:
            a list with temperature boundary conditions

        """

        def full_boundary(x):
            if self.p["dim"] == 2:
                return (
                    self.boundary_bottom()(x)
                    | self.boundary_left()(x)
                    | self.boundary_right()(x)
                    | self.boundary_top()(x)
                )
            elif self.p["dim"] == 3:
                return (
                    self.boundary_back()(x)
                    | self.boundary_bottom()(x)
                    | self.boundary_front()(x)
                    | self.boundary_left()(x)
                    | self.boundary_right()(x)
                    | self.boundary_top()(x)
                )

        bc_generator = BoundaryConditions(self.mesh, V)
        bc_generator.add_dirichlet_bc(
            self.temperature_bc,
            boundary=full_boundary,
            method="geometrical",
            entity_dim=self.mesh.topology.dim - 1,
        )
        return bc_generator.bcs

    def create_body_force(self, v: ufl.argument.Argument) -> ufl.form.Form | None:
        # TODO: The sign of the body force is not clear.

        if self.use_body_force:
            force_vector = np.zeros(self.p["dim"])
            force_vector[-1] = self.p["rho"] * self.p["g"]  # works for 2D and 3D

            f = df.fem.Constant(self.mesh, force_vector)
            L = ufl.dot(f, v) * ufl.dx

            return L
        else:
            return None
