import logging

import dolfinx as df
import numpy as np
import pint
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType

from fenicsxconcrete.experimental_setup.am_multiple_layers import AmMultipleLayers
from fenicsxconcrete.experimental_setup.base_experiment import Experiment
from fenicsxconcrete.finite_element_problem.base_material import MaterialProblem
from fenicsxconcrete.helper import Parameters
from fenicsxconcrete.unit_registry import ureg


class ConcreteAM(MaterialProblem):
    """
    A class for additive manufacturing model
    - including pseudo density approach for element activation -> set_initial_path == negative time when dof will be activated
    - incremental weak form (in case of density load increments are computed automatic, otherwise user controlled)
    - possible crresponding material laws
        - [concretethixelasticmodel] linear elastic thixotropy = linear elastic with age dependent Young's modulus
        - [concreteviscodevthixelasticmodel] thixotropy-viscoelastic model (Three parameter model: CMaxwell or CKelvin) with deviator assumption with age dependent moduli
        - ...

    Args:
        experiment: The experimental setup.
        parameters: Dictionary with parameters.
        material_law: string of class name of used material law (in the moment: concretethixelasticmodel or concreteviscodevthixelasticmodel)
        pv_name: Name of the paraview file, if paraview output is generated.
        pv_path: Name of the paraview path, if paraview output is generated.
    """

    def __init__(
        self,
        experiment: Experiment,
        parameters: dict[str, pint.Quantity],
        material_law: str,
        pv_name: str = "pv_output_full",
        pv_path: str | None = None,
    ) -> None:

        # adding default material parameter, will be overridden by outside input
        default_p = Parameters()

        # updating parameters, overriding defaults
        default_p.update(parameters)

        # set name of material law class
        self.material_law = material_law

        self.logger(__name__)
        super().__init__(experiment, default_p, pv_name, pv_path)

    @staticmethod
    def default_parameters() -> tuple[Experiment, dict[str, pint.Quantity]]:
        """
        Static method that returns a set of default parameters.
        Returns:
            The default parameters as a dictionary.
        """
        # default experiment
        experiment = AmMultipleLayers(AmMultipleLayers.default_parameters())

        model_parameters = {}
        # Material parameter for concrete model with structural build-up
        model_parameters["rho"] = 2070 * ureg("kg/m^3")  # density of fresh concrete
        model_parameters["nu"] = 0.3 * ureg("")  # Poissons Ratio

        ### default parameters required for thix elastic model
        # Youngs modulus is changing over age (see E_fkt) following the bilinear approach Kruger et al 2019
        # (https://www.sciencedirect.com/science/article/pii/S0950061819317507) with two different rates
        model_parameters["E_0"] = 15000 * ureg("Pa")  # Youngs Modulus at age=0
        model_parameters["R_E"] = 15 * ureg("Pa/s")  # Reflocculation (first) rate
        model_parameters["A_E"] = 30 * ureg("Pa/s")  # Structuration (second) rate
        model_parameters["t_f"] = 300 * ureg("s")  # Reflocculation time (switch point)
        model_parameters["age_0"] = 0 * ureg("s")  # start age of concrete

        # other model parameters
        model_parameters["degree"] = 2 * ureg("")  # polynomial degree
        model_parameters["u_bc"] = 0.1 * ureg("")  # displacement on top
        model_parameters["load_time"] = 1 * ureg("s")  # load applied in 1 s

        return experiment, model_parameters

    def setup(self) -> None:
        # set up problem

        if self.mech_prob_string.lower() == "concretethixelasticmodel":
            self.mechanics_problem = ConcreteThixElasticModel(self.experiment.mesh, self.p, pv_name=self.pv_name)
        else:
            raise ValueError(f"material law {self.mechanics_problem} not yet implemented")

        self.V = self.mechanics_problem.V  # for reaction force sensor
        self.residual = None  # initialize

        # setting bcs
        bcs = self.experiment.create_displacement_boundary(self.mechanics_problem.V)
        # external load
        external_force = self.experiment.create_force_boundary(self.v)
        # body load
        body_force = self.experiment.create_body_force(self.v)
        self.mechanics_problem.set(bcs, external_force, body_force)

        # setting up the solver
        self.mechanics_solver = df.nls.petsc.NewtonSolver(MPI.COMM_WORLD, self.mechanics_problem)
        self.mechanics_solver.convergence_criterion = "incremental"
        self.mechanics_solver.atol = 1e-7
        self.mechanics_solver.rtol = 1e-6
        self.mechanics_solver.report = True

    def solve(self, t: float = 1.0) -> None:
        # define what to do, to solve this problem
        self.logger.info(f"solve for t:{t}")

        # CHANGED FOR INCREMENTAL SET UP from u to du!!!
        self.mechanics_solver.solve(self.mechanics_problem.du)

        # save fields to global problem for sensor output
        self.displacement = self.mechanics_problem.u
        self.stress = self.mechanics_problem.q_sig
        self.strain = self.mechanics_problem.q_eps
        # general interface if stress/strain are in voigt or full tensor format is specified in mechanics_problem!!
        self.visu_space_stress = self.mechanics_problem.visu_space_sig
        self.visu_space_strain = self.mechanics_problem.visu_space_eps

        # get sensor data
        self.compute_residuals()  # for residual sensor
        for sensor_name in self.sensors:
            # go through all sensors and measure
            self.sensors[sensor_name].measure(self, t)

        # update age & path before next step!
        self.mechanics_problem.update_values()

    def compute_residuals(self) -> None:
        # define what to do, to compute the residuals. Called in solve
        self.residual = ufl.action(self.mecahnics_problem.R, self.displacement)

    def set_initial_path(self, path):
        self.mechanics_problem.set_initial_path(path)


class ConcreteThixElasticModel(df.fem.petsc.NonlinearProblem):
    # linear elasticity law with age dependent Youngs modulus modelling the thixotropy
    # tensor format!!
    # incremental formulated u = u_old + du solve for du with given load increment df (using function q_fd) and material evaluation for dsigma/depsilon

    def __init__(self, mesh, p, pv_name="mechanics_output", **kwargs):
        super().__init__(self)

    def F(self, x: PETSc.Vec, b: PETSc.Vec):
        super().F(x, b)

    def J(self, x: PETSc.Vec, A: PETSc.Mat):
        super().J(x, A)
