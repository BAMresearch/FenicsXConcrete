from collections.abc import Callable

import dolfinx as df
import numpy as np
import pint
import ufl
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from fenicsxconcrete.boundary_conditions.bcs import BoundaryConditions
from fenicsxconcrete.experimental_setup.base_experiment import Experiment
from fenicsxconcrete.util import Parameters, QuadratureRule, ureg


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

        # initialize a set of default parameters
        default_p = Parameters()
        default_p["height"] = 1 * ureg("m")
        default_p["width"] = 1 * ureg("m")
        default_p["length"] = 1 * ureg("m")

        default_p.update(parameters)

        super().__init__(default_p)

    @staticmethod
    def default_parameters() -> dict[str, pint.Quantity]:
        """sets up a working set of parameter values as example

        Returns:
            dictionary with a working set of the required parameter

        """

        setup_parameters = {}

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
                # displacement controlled in one direction
                bc_generator.add_dirichlet_bc(
                    self.top_displacement, boundary=self.boundary_top(), sub=1, method="geometrical", entity_dim=1
                )
            elif self.p["strain_state"] == "multiaxial":
                # displacement controlled in all directions
                bc_generator.add_dirichlet_bc(
                    self.top_displacement, boundary=self.boundary_top(), sub=1, method="geometrical", entity_dim=1
                )
                bc_generator.add_dirichlet_bc(
                    self.top_displacement, boundary=self.boundary_right(), sub=0, method="geometrical", entity_dim=1
                )
            elif self.p["strain_state"] == "stress_controlled":
                pass
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

            # displacement controlled in one direction
            if self.p["strain_state"] == "uniaxial":
                bc_generator.add_dirichlet_bc(
                    self.top_displacement, boundary=self.boundary_top(), sub=2, method="geometrical", entity_dim=2
                )
            elif self.p["strain_state"] == "multiaxial":
                # displacement controlled in all directions
                bc_generator.add_dirichlet_bc(
                    self.top_displacement, boundary=self.boundary_top(), sub=2, method="geometrical", entity_dim=2
                )
                bc_generator.add_dirichlet_bc(
                    self.top_displacement, boundary=self.boundary_right(), sub=0, method="geometrical", entity_dim=2
                )
                bc_generator.add_dirichlet_bc(
                    self.top_displacement, boundary=self.boundary_back(), sub=1, method="geometrical", entity_dim=2
                )
            elif self.p["strain_state"] == "stress_controlled":
                pass
            else:
                raise ValueError(f'Strain_state value: {self.p["strain_state"]} is not implemented in 3D.')

        return bc_generator.bcs

    def apply_displ_load(self, top_displacement: pint.Quantity | float) -> None:
        """Updates the applied displacement load

        Args:
            top_displacement: Displacement of the top boundary in mm, > 0 ; tension, < 0 ; compression

        """
        top_displacement.to_base_units()
        self.top_displacement.value = top_displacement.magnitude

    def create_body_force_am(
        self, v: ufl.argument.Argument, q_fd: df.fem.Function, rule: QuadratureRule
    ) -> ufl.form.Form:
        """defines body force for am experiments

        element activation via pseudo density and incremental loading via parameter ["load_time"] computed in class concrete_am

        Args:
            v: test function
            q_fd: quadrature function given the loading increment where elements are active

        Returns:
            form for body force

        """
        print("in create body force am")

        force_vector = np.zeros(self.p["dim"])
        force_vector[-1] = -self.p["rho"] * self.p["g"]  # works for 2D and 3D
        print(force_vector)
        f = df.fem.Constant(self.mesh, ScalarType(force_vector))
        L = q_fd * ufl.dot(f, v) * rule.dx

        return L
