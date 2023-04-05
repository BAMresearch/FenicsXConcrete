import os
from collections.abc import Callable

import dolfinx as df
import meshio
import numpy as np
import pint
import ufl
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from fenicsxconcrete.boundary_conditions.bcs import BoundaryConditions
from fenicsxconcrete.boundary_conditions.boundary import plane_at
from fenicsxconcrete.experimental_setup.base_experiment import Experiment
from fenicsxconcrete.helper import Parameters
from fenicsxconcrete.unit_registry import ureg


class MinimalCubeExperiment(Experiment):
    """A cylinder mesh for a uni-axial displacement load"""

    def __init__(self, parameters: dict[str, pint.Quantity] | None = None) -> None:
        """initializes the object

        Standard parameters are set
        setup function called

        Parameters
        ----------
        parameters : dictionary with parameters that can override the default values
        """
        # initialize a set of default parameters
        default_p = Parameters()
        default_p["T_0"] = 20.0 * ureg.dC

        default_p.update(parameters)

        super().__init__(default_p)

    def setup(self) -> None:
        """Generates the mesh based on parameters

        This function is called during __init__
        """

        if self.p["dim"] == 2:
            # build a rectangular mesh
            self.mesh = df.mesh.create_rectangle(
                MPI.COMM_WORLD,
                np.vstack(
                    (
                        self.p["start_point"],
                        self.p["end_point"],
                    )
                ),
                self.p["num_elements"],
                cell_type=df.mesh.CellType.quadrilateral,
            )
        elif self.p["dim"] == 3:
            self.mesh = df.mesh.create_box(
                MPI.COMM_WORLD,
                np.vstack(
                    (
                        self.p["start_point"],
                        self.p["end_point"],
                    )
                ),
                self.p["num_elements"],
                cell_type=df.mesh.CellType.hexahedron,
            )

        else:
            raise ValueError(f"wrong dimension {self.p['dim']} for problem setup")

    @staticmethod
    def default_parameters() -> dict[str, pint.Quantity]:
        """
        Resturns: A dictionary with required parameters and a set of working values as example.
        """
        # this must de defined in each setup class

        setup_parameters = {
            "dim": 3 * ureg(""),
            "start_point": np.array([0.0, 0.0, 0.0]) * ureg("m"),
            "end_point": np.array([1.0, 1.0, 1.0]) * ureg("m"),
            "num_elements": np.array([10, 10, 10]) * ureg("m"),
        }
        return setup_parameters

    def create_displacement_boundary(self, V: df.fem.FunctionSpace) -> list[df.fem.bcs.DirichletBCMetaClass]:
        """Defines the displacement boundary conditions

        Args:
            V :Function space of the structure

        Returns:
            displ_bc :A list of DirichletBC objects, defining the boundary conditions
        """

        # define boundary conditions generator
        bc_generator = BoundaryConditions(self.mesh, V)
        bc_generator.add_dirichlet_bc(np.zeros(self.p["dim"]), self.boundary_bottom(), method="geometrical")

        return bc_generator.bcs

    def create_temperature_bcs(self, V: df.fem.FunctionSpace):

        return []

    def apply_displ_load(self, top_displacement: pint.Quantity | float) -> None:
        """Updates the applied displacement load

        Parameters
        ----------
        top_displacement : Displacement of the top boundary in mm, > 0 ; tension, < 0 ; compression
        """

        self.top_displacement.value = top_displacement.magnitude

    def create_body_force(self, v: ufl.argument.Argument) -> ufl.form.Form:
        force_vector = np.zeros(self.p["dim"])
        force_vector[-1] = -self.p["rho"] * self.p["g"]  # works for 2D and 3D

        f = df.fem.Constant(self.mesh, ScalarType(force_vector))
        L = ufl.dot(f, v) * ufl.dx

        return L

    def boundary_top(self) -> Callable:
        if self.p["dim"] == 2:
            return plane_at(self.p["height"], 1)
        elif self.p["dim"] == 3:
            return plane_at(self.p["height"], 2)

    def boundary_bottom(self) -> Callable:
        if self.p["dim"] == 2:
            return plane_at(0.0, 1)
        elif self.p["dim"] == 3:
            return plane_at(0.0, 2)
