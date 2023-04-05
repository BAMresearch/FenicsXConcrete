from collections.abc import Callable

import dolfinx as df
import numpy as np
import pint
from mpi4py import MPI

from fenicsxconcrete.boundary_conditions.bcs import BoundaryConditions
from fenicsxconcrete.boundary_conditions.boundary import plane_at
from fenicsxconcrete.experimental_setup.base_experiment import Experiment
from fenicsxconcrete.helper import Parameters
from fenicsxconcrete.unit_registry import ureg


class UniaxialCubeExperiment(Experiment):
    """set up an uniaxial cube structure with displacement load

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

        default_p.update(parameters)

        super().__init__(default_p)

    @staticmethod
    def default_parameters() -> dict[str, pint.Quantity]:
        """set up a working set of parameter values as example

        Returns:
            dictionary with a working set of the required parameter

        """

        setup_parameters = {}

        setup_parameters["dim"] = 3 * ureg("")
        setup_parameters["num_elements_length"] = 2 * ureg("")
        setup_parameters["num_elements_width"] = 2 * ureg("")
        setup_parameters["num_elements_height"] = 2 * ureg("")

        return setup_parameters

    def setup(self) -> None:
        """Generates the mesh in 2D or 3D based on parameters"""

        self.logger.debug("setup mesh for %s", self.p["dim"])

        if self.p["dim"] == 2:
            # build a rectangular mesh
            self.mesh = df.mesh.create_rectangle(
                MPI.COMM_WORLD,
                [
                    [0.0, 0.0],
                    [1.0, 1.0],
                ],
                [self.p["num_elements_length"], self.p["num_elements_height"]],
                cell_type=df.mesh.CellType.quadrilateral,
            )
        elif self.p["dim"] == 3:
            self.mesh = df.mesh.create_box(
                MPI.COMM_WORLD,
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
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
                np.float64(0.0), boundary=self.boundary_left(), sub=0, method="geometrical", entity_dim=0
            )

            # displacement controlled
            bc_generator.add_dirichlet_bc(
                self.top_displacement, boundary=self.boundary_top(), sub=1, method="geometrical", entity_dim=1
            )

        elif self.p["dim"] == 3:
            # uniaxial bcs
            bc_generator.add_dirichlet_bc(
                np.float64(0.0), boundary=self.boundary_bottom(), sub=2, method="geometrical", entity_dim=2
            )
            bc_generator.add_dirichlet_bc(
                np.float64(0.0), boundary=self.boundary_left(), sub=0, method="geometrical", entity_dim=0
            )
            bc_generator.add_dirichlet_bc(
                np.float64(0.0), boundary=self.boundary_front(), sub=1, method="geometrical", entity_dim=1
            )

            # displacement controlled
            bc_generator.add_dirichlet_bc(
                self.top_displacement, boundary=self.boundary_top(), sub=2, method="geometrical", entity_dim=2
            )

        return bc_generator.bcs

    def apply_displ_load(self, top_displacement: pint.Quantity | float) -> None:
        """Updates the applied displacement load

        Args:
            top_displacement: Displacement of the top boundary in mm, > 0 ; tension, < 0 ; compression

        """

        self.top_displacement.value = top_displacement.magnitude

    def boundary_bottom(self) -> Callable:
        """specify boundary: plane at bottom

        Returns:
            fct defining if dof is at boundary

        """

        if self.p["dim"] == 2:
            return plane_at(0.0, "y")
        elif self.p["dim"] == 3:
            return plane_at(0.0, "z")

    def boundary_left(self) -> Callable:
        """specify boundary: plane at left side

        Returns:
            fct defining if dof is at boundary

        """

        if self.p["dim"] == 2:
            return plane_at(0.0, "x")
        elif self.p["dim"] == 3:
            return plane_at(0.0, "x")

    def boundary_top(self) -> Callable:
        """specify boundary: plane at top

        Returns:
            fct defining if dof is at boundary

        """

        if self.p["dim"] == 2:
            return plane_at(1.0, "y")
        elif self.p["dim"] == 3:
            return plane_at(1.0, "z")

    def boundary_front(self) -> Callable:
        """specify boundary: plane at front

        only for 3D case front plane

        Returns:
            fct defining if dof is at boundary

        """

        if self.p["dim"] == 3:
            return plane_at(0.0, "y")
