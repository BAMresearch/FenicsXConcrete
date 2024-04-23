import copy
from collections.abc import Callable
from typing import Type

import basix
import dolfinx as df
import numpy as np
import pint
import ufl
from fenics_constitutive import (
    Constraint,
    IncrSmallStrainModel,
    IncrSmallStrainProblem,
    build_history,
    ufl_mandel_strain,
)
from mpi4py import MPI
from petsc4py import PETSc

from fenicsxconcrete.experimental_setup import Experiment, SimpleCube
from fenicsxconcrete.finite_element_problem.base_material import MaterialProblem, QuadratureFields, SolutionFields
from fenicsxconcrete.util import Parameters, QuadratureEvaluator, QuadratureRule, project, ureg


class FenicsConstitutive(MaterialProblem):
    """A class for material laws defined in the fenics-constitutive interface

    - connects material problem class with incremental small strain problem from fenics-constitutive

    Attributes:
        material: the IncrSmallStrainModel given the material law
        further: see base class
    """

    def __init__(
        self,
        experiment: Experiment,
        parameters: dict[str, pint.Quantity],
        material: IncrSmallStrainModel,
        pv_name: str = "pv_output_full",
        pv_path: str | None = None,
    ) -> None:
        """initialize object

        Args:
            experiment: The experimental setup.
            parameters: Dictionary with parameters.
            material: material law as IncSmallStrainModel from fenics-constitutive
            pv_name: Name of the paraview file, if paraview output is generated.
            pv_path: Name of the paraview path, if paraview output is generated.

        """

        if material:
            self.material_law = material
        else:
            raise ValueError("material law not supported")

        super().__init__(experiment, parameters, pv_name, pv_path)

    @staticmethod
    def parameter_description() -> dict[str, str]:
        """static method returning a description dictionary for required parameters

        Returns:
            description dictionary

        """
        description = {
            "rho": "Density of material",
            "g": "Gravitational acceleration",
            "degree": "Polynomial degree for the FEM model",
            "q_degree": "Polynomial degree for which the quadrature rule integrates correctly",
            "dt": "time step",
            "material parameters": "select according to chosen material law!",
        }

        return description

    @staticmethod
    def default_parameters() -> tuple[Experiment, dict[str, pint.Quantity]]:
        """Static method that returns a set of default parameters for the selected nonlinear problem.

        Returns:
            The default experiment instance and the default parameters as a dictionary.

        """

        # default experiment
        experiment = SimpleCube(SimpleCube.default_parameters())

        # default parameters according given nonlinear problem #TODO
        parameters = {
            # general parameters
            "rho": 2070 * ureg("kg/m^3"),  # density
            "g": 9.81 * ureg("m/s^2"),  # gravity
            # general model parameters
            "degree": 2 * ureg(""),  # polynomial degree
            "q_degree": 2 * ureg(""),  # quadrature rule
            "dt": 1.0 * ureg("s"),  # time step
            # material parameters
            # ... - according to chosen material law!
        }

        return experiment, {**parameters}

    def setup(self) -> None:
        """set up problem"""

        # displacement space and field
        self.V = df.fem.VectorFunctionSpace(self.experiment.mesh, ("CG", self.p["degree"]))
        self.fields = SolutionFields(displacement=df.fem.Function(self.V, name="displacement"))

        # define problem:

        # material law based on fenics constitutive interface
        law = self.material_law(self.p, constraint=Constraint.FULL)

        # boundaries
        bcs = self.experiment.create_displacement_boundary(self.V)
        # body_force_fct = self.experiment.create_body_force # not yet in IncrSmallStrainProblem

        # problem
        self.mechanics_problem = IncrSmallStrainProblem(
            law, self.fields.displacement, bcs, q_degree=self.p["q_degree"]
        )

        # additional output fields
        self.rule = QuadratureRule(cell_type=self.mesh.ufl_cell(), degree=self.p["q_degree"])
        self.strain_stress_space = self.rule.create_quadrature_tensor_space(self.mesh, (self.p["dim"], self.p["dim"]))
        self.q_fields = QuadratureFields(
            measure=self.rule.dx,
            plot_space_type=("DG", self.p["degree"] - 1),
            strain=df.fem.Function(self.strain_stress_space, name="strain"),
            stress=df.fem.Function(self.strain_stress_space, name="stress"),
        )

        # setting up the solver
        self.mechanics_solver = df.nls.petsc.NewtonSolver(MPI.COMM_WORLD, self.mechanics_problem)
        self.mechanics_solver.atol = 1e-9
        self.mechanics_solver.rtol = 1e-8
        self.mechanics_solver.report = True

    def solve(self) -> None:
        """time incremental solving !"""

        self.update_time()  # set t+dt

        self.logger.info(f"solve for t: {self.time}")
        self.logger.info(f"CHECK if external loads are applied as incremental loads e.g. delta_u(t)!!!")

        # solve problem for current time increment
        n, converged = self.mechanics_solver.solve(self.fields.displacement)
        if not converged:
            self.logger.warning("Mechanics solve did not converge")
        else:
            self.logger.info(f"Mechanics solve converged in {n} iterations")

        self.mechanics_problem.update()  # TODO at which point?

        # save fields to global problem for sensor output
        # TODO

        # get sensor data
        self.compute_residuals()  # for residual sensor
        for sensor_name in self.sensors:
            # go through all sensors and measure
            self.sensors[sensor_name].measure(self)

    def compute_residuals(self) -> None:
        """defines what to do, to compute the residuals. Called in solve for sensors"""

        self.residual = self.mechanics_problem.R_form

    def pv_plot(self) -> None:
        """creates paraview output at given time step"""

        self.logger.info(f"create pv plot for t: {self.time}")

        # # write further fields
        # sigma_plot = project(
        #     self.mechanics_problem.sigma(self.fields.displacement),
        #     df.fem.TensorFunctionSpace(self.mesh, self.q_fields.plot_space_type),
        #     self.rule.dx,
        # )
        #
        # E_plot = project(
        #     self.mechanics_problem.q_E, df.fem.FunctionSpace(self.mesh, self.q_fields.plot_space_type), self.rule.dx
        # )
        #
        # E_plot.name = "Youngs_Modulus"
        # sigma_plot.name = "Stress"
        #
        with df.io.XDMFFile(self.mesh.comm, self.pv_output_file, "a") as f:
            f.write_function(self.fields.displacement, self.time)
        #     f.write_function(sigma_plot, self.time)
        #     f.write_function(E_plot, self.time)
        #
