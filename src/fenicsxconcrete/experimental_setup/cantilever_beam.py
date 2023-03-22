import dolfinx as df
import numpy as np
import pint
import ufl
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from fenicsxconcrete.experimental_setup.base_experiment import Experiment
from fenicsxconcrete.helper import Parameters
from fenicsxconcrete.unit_registry import ureg


class CantileverBeam(Experiment):
    """Sets up a cantilever beam, clamped on one side and loaded with gravity

    Attributes:
        see base class
    """

    def __init__(self, parameters: dict[str, pint.Quantity] | None = None):
        """defines default parameters, for the rest, see base class"""

        # initialize default parameters for the setup
        default_p = Parameters()
        # default_p['dummy'] = 'example' * ureg('')  # example default parameter for this class

        # updating parameters, overriding defaults
        default_p.update(parameters)

        super().__init__(default_p)

    def setup(self) -> None:
        """defines the mesh for 2D or 3D"""

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
        """returns a dictionary with required parameters and a set of working values as example"""
        # this must de defined in each setup class

        setup_parameters = {}
        setup_parameters["length"] = 1 * ureg("m")
        setup_parameters["height"] = 0.3 * ureg("m")
        setup_parameters["width"] = 0.3 * ureg("m")  # only relevant for 3D case
        setup_parameters["dim"] = 3 * ureg("")
        setup_parameters["num_elements_length"] = 10 * ureg("")
        setup_parameters["num_elements_height"] = 3 * ureg("")
        setup_parameters["num_elements_width"] = 3 * ureg("")  # only relevant for 3D case

        return setup_parameters

    def create_displacement_boundary(self, V: df.fem.FunctionSpace) -> list[df.fem.bcs.DirichletBCMetaClass]:
        # define displacement boundary

        # fenics will individually call this function for every node and will note the true or false value.
        def clamped_boundary(x):
            return np.isclose(x[0], 0)

        displacement_bcs = []

        zero = np.zeros(self.p["dim"])
        displacement_bcs.append(
            df.fem.dirichletbc(
                np.array(zero, dtype=ScalarType),
                df.fem.locate_dofs_geometrical(V, clamped_boundary),
                V,
            )
        )

        return displacement_bcs

    def create_body_force(self, v: ufl.argument.Argument) -> ufl.form.Form:
        force_vector = np.zeros(self.p["dim"])
        force_vector[-1] = -self.p["rho"] * self.p["g"]  # works for 2D and 3D

        f = df.fem.Constant(self.mesh, ScalarType(force_vector))
        L = ufl.dot(f, v) * ufl.dx

        return L
