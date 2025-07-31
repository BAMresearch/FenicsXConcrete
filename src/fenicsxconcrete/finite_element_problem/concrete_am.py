import copy
from collections.abc import Callable
from typing import Type

import dolfinx as df
import numpy as np
import pint
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from fenicsxconcrete.experimental_setup import AmMultipleLayers, Experiment
from fenicsxconcrete.finite_element_problem.base_material import MaterialProblem, QuadratureFields, SolutionFields
from fenicsxconcrete.util import Parameters, QuadratureEvaluator, QuadratureRule, project, ureg


class ConcreteAM(MaterialProblem):
    """A class for additive manufacturing models

    - including pseudo density approach for element activation -> set_initial_path == negative time when element will be activated
    - time incremental weak form (in case of density load increments are computed automatic, otherwise user controlled)
    - possible corresponding material laws
        - [concretethixelasticmodel] linear elastic thixotropy = linear elastic with age dependent Young's modulus
        - [concreteviscodevthixelasticmodel] thixotropy-viscoelastic model (Three parameter model: CMaxwell or CKelvin) with deviator assumption with age dependent moduli
        - ...

    Attributes:
        nonlinear_problem: the nonlinear problem class of used material law
        further: see base class
    """

    def __init__(
        self,
        experiment: Experiment,
        parameters: dict[str, pint.Quantity],
        nonlinear_problem: Type[df.fem.petsc.NonlinearProblem] | None = None,
        pv_name: str = "pv_output_full",
        pv_path: str | None = None,
    ) -> None:
        """initialize object

        Args:
            experiment: The experimental setup.
            parameters: Dictionary with parameters.
            nonlinear_problem: the nonlinear problem class of used material law
            pv_name: Name of the paraview file, if paraview output is generated.
            pv_path: Name of the paraview path, if paraview output is generated.

        """

        # # adding default material parameter, will be overridden by outside input
        # default_p = Parameters()
        #  # default stress state for 2D optional "plane_stress"
        #
        # # updating parameters, overriding defaults
        # default_p.update(parameters)

        if nonlinear_problem:
            self.nonlinear_problem = nonlinear_problem
        else:
            self.nonlinear_problem = ConcreteThixElasticModel  # default material

        super().__init__(experiment, parameters, pv_name, pv_path)

    @staticmethod
    def parameter_description() -> dict[str, str]:
        """static method returning a description dictionary for required parameters

        Returns:
            description dictionary

        """
        description = {
            "general parameters": {
                "rho": "density of fresh concrete",
                "g": "gravity",
                "nu": "Poissons Ratio",
                "degree": "Polynomial degree for the FEM model",
                "q_degree": "Polynomial degree for which the quadrature rule integrates correctly",
                "load_time": "time in which the load is applied",
                "stress_state": "for 2D plain stress or plane strain",
                "dt": "time step",
            },
            "ThixElasticModel": {
                "E_0": "Youngs Modulus at age=0",
                "R_E": "Reflocculation (first) rate",
                "A_E": "Structuration (second) rate",
                "tf_E": "Reflocculation time (switch point)",
                "age_0": "Start age of concrete",
            },
        }

        return description

    @staticmethod
    def default_parameters(
        non_linear_problem: df.fem.petsc.NonlinearProblem | None = None,
    ) -> tuple[Experiment, dict[str, pint.Quantity]]:
        """Static method that returns a set of default parameters for the selected nonlinear problem.

        Args:
            non_linear_problem: the nonlinear problem class of used material law

        Returns:
            The default experiment instance and the default parameters as a dictionary.

        """

        # default experiment
        experiment = AmMultipleLayers(AmMultipleLayers.default_parameters())

        # default parameters according given nonlinear problem
        joined_parameters = {
            # Material parameter for concrete model with structural build-up
            "rho": 2070 * ureg("kg/m^3"),  # density of fresh concrete
            "g": 9.81 * ureg("m/s^2"),  # gravity
            "nu": 0.3 * ureg(""),  # Poissons Ratio
            # other model parameters
            "degree": 2 * ureg(""),  # polynomial degree
            "q_degree": 2 * ureg(""),  # quadrature rule
            "stress_state": "plane_strain" * ureg(""),  # for 2D stress state
            "dt": 1.0 * ureg("s"),  # time step
            "load_time": 60 * ureg("s"),  # body force load applied in s
        }
        if not non_linear_problem or non_linear_problem == ConcreteThixElasticModel:
            ### default parameters required for ThixElasticModel
            model_parameters = {
                # Youngs modulus is changing over age (see E_fkt) following the bilinear approach Kruger et al 2019
                # (https://www.sciencedirect.com/science/article/pii/S0950061819317507) with two different rates
                "E_0": 15000 * ureg("Pa"),  # Youngs Modulus at age=0
                "R_E": 15 * ureg("Pa/s"),  # Reflocculation (first) rate
                "A_E": 30 * ureg("Pa/s"),  # Structuration (second) rate
                "tf_E": 300 * ureg("s"),  # Reflocculation time (switch point)
                "age_0": 0 * ureg("s"),  # start age of concrete
            }
        else:
            raise ValueError("non_linear_problem not supported")

        return experiment, {**joined_parameters, **model_parameters}

    def setup(self) -> None:
        """set up problem"""

        self.rule = QuadratureRule(cell_type=self.mesh.ufl_cell(), degree=self.p["q_degree"])
        # displacement space (name V required for sensors!)
        self.V = df.fem.VectorFunctionSpace(self.mesh, ("CG", self.p["degree"]))
        self.strain_stress_space = self.rule.create_quadrature_tensor_space(self.mesh, (self.p["dim"], self.p["dim"]))

        # global variables for all AM problems relevant
        self.fields = SolutionFields(displacement=df.fem.Function(self.V, name="displacement"))

        self.q_fields = QuadratureFields(
            measure=self.rule.dx,
            plot_space_type=("DG", self.p["degree"] - 1),
            strain=df.fem.Function(self.strain_stress_space, name="strain"),
            stress=df.fem.Function(self.strain_stress_space, name="stress"),
        )

        # displacement increment
        self.d_disp = df.fem.Function(self.V)

        # boundaries
        bcs = self.experiment.create_displacement_boundary(self.V)
        body_force_fct = self.experiment.create_body_force_am

        self.mechanics_problem = self.nonlinear_problem(
            self.mesh,
            self.p,
            self.rule,
            self.d_disp,
            bcs,
            body_force_fct,
        )

        # setting up the solver
        self.mechanics_solver = df.nls.petsc.NewtonSolver(MPI.COMM_WORLD, self.mechanics_problem)
        self.mechanics_solver.convergence_criterion = "incremental"
        self.mechanics_solver.atol = 1e-9
        self.mechanics_solver.rtol = 1e-8
        self.mechanics_solver.report = True

    def solve(self) -> None:
        """time incremental solving !"""

        self.update_time()  # set t+dt
        self.update_path()  # set path

        self.logger.info(f"solve for t: {self.time}")
        self.logger.info(f"CHECK if external loads are applied as incremental loads e.g. delta_u(t)!!!")

        # solve problem for current time increment
        self.mechanics_solver.solve(self.d_disp)

        # update total displacement
        self.fields.displacement.vector.array[:] += self.d_disp.vector.array[:]
        self.fields.displacement.x.scatter_forward()

        # save fields to global problem for sensor output
        self.q_fields.stress.vector.array[:] += self.mechanics_problem.q_sig.vector.array[:]
        self.q_fields.stress.x.scatter_forward()
        self.q_fields.strain.vector.array[:] += self.mechanics_problem.q_eps.vector.array[:]
        self.q_fields.strain.x.scatter_forward()

        # additional output field not yet used in any sensors
        self.youngsmodulus = self.mechanics_problem.q_E

        # get sensor data
        self.compute_residuals()  # for residual sensor
        for sensor_name in self.sensors:
            # go through all sensors and measure
            self.sensors[sensor_name].measure(self)

        # update path & internal variables before next step!
        self.mechanics_problem.update_history(fields=self.fields, q_fields=self.q_fields)  # if required otherwise pass

    def compute_residuals(self) -> None:
        """defines what to do, to compute the residuals. Called in solve for sensors"""

        self.residual = self.mechanics_problem.R

    def update_path(self) -> None:
        """update path for next time increment"""
        self.mechanics_problem.q_array_path += self.p["dt"] * np.ones_like(self.mechanics_problem.q_array_path)

    def set_initial_path(self, path: list[float] | float) -> None:
        """set initial path for problem

        Args:
            path: array describing the negative time when an element will be reached on quadrature space
                    if only one value is given, it is assumed that all elements are reached at the same time

        """
        if isinstance(path, float):
            self.mechanics_problem.q_array_path = path * np.ones_like(self.mechanics_problem.q_array_path)
        else:
            self.mechanics_problem.q_array_path = path

    def pv_plot(self) -> None:
        """creates paraview output at given time step"""

        self.logger.info(f"create pv plot for t: {self.time}")

        # write further fields
        sigma_plot = project(
            self.mechanics_problem.sigma(self.fields.displacement),
            df.fem.TensorFunctionSpace(self.mesh, self.q_fields.plot_space_type),
            self.rule.dx,
        )

        E_plot = project(
            self.mechanics_problem.q_E, df.fem.FunctionSpace(self.mesh, self.q_fields.plot_space_type), self.rule.dx
        )

        E_plot.name = "Youngs_Modulus"
        sigma_plot.name = "Stress"

        with df.io.XDMFFile(self.mesh.comm, self.pv_output_file, "a") as f:
            f.write_function(self.fields.displacement, self.time)
            f.write_function(sigma_plot, self.time)
            f.write_function(E_plot, self.time)

    @staticmethod
    def fd_fkt(pd: list[float], path_time: list[float], dt: float, load_time: float) -> list[float]:
        """computes weighting fct for body force term in pde

        body force can be applied in several loading steps given by parameter ["load_time"]
        load factor for each step = 1 / "load_time" * dt
        can be used in all nonlinear problems

        Args:
            pd: array of pseudo density values
            path_time: array of process time values
            dt: time step value
            load_time: time when load is fully applied

        Returns:
            array of incremental weigths for body force
        """
        fd = np.zeros_like(pd)

        active_idx = np.where(pd > 0)[0]  # only active elements
        # select indices where path_time is smaller than load_time and bigger then zero [since usually we start the computation at dt so that also for further layers the laoding starts at local layer time +dt]
        load_idx = np.where((path_time[active_idx] <= load_time) & (path_time[active_idx] > 0))
        for _ in load_idx:
            fd[active_idx[load_idx]] = dt / load_time  # linear ramp

        return fd

    @staticmethod
    def pd_fkt(path_time: list[float]) -> list[float]:
        """computes pseudo density array

        pseudo density: decides if layer is there (age >=0 active) or not (age < 0 nonactive!)
        decision based on current path_time value
        can be used in all nonlinear problems

        Args:
            path_time: array of process time values at quadrature points

        Returns:
            array of pseudo density
        """

        l_active = np.zeros_like(path_time)  # 0: non-active

        activ_idx = np.where(path_time >= 0 - 1e-5)[0]
        l_active[activ_idx] = 1.0  # active

        return l_active

    @staticmethod
    def E_fkt(pd: float, path_time: float, parameters: dict) -> float:
        """computes the Young's modulus at current quadrature point according to bilinear Kruger model

        Args:
            pd: value of pseudo density [0 - non active or 1 - active]
            path_time: process time value
            parameters: parameter dict for bilinear model described by (P0,R_P,A_P,tf_P,age_0)

        Returns:
            value of current Young's modulus
        """
        # print(parameters["age_0"] + path_time)
        if pd > 0:  # element active, compute current Young's modulus
            age = parameters["age_0"] + path_time  # age concrete
            if age < parameters["tf_P"]:
                E = parameters["P0"] + parameters["R_P"] * age
            elif age >= parameters["tf_P"]:
                E = (
                    parameters["P0"]
                    + parameters["R_P"] * parameters["tf_P"]
                    + parameters["A_P"] * (age - parameters["tf_P"])
                )
        else:
            E = 1e-4  # non-active

        return E


class ConcreteThixElasticModel(df.fem.petsc.NonlinearProblem):
    """linear elastic thixotropy concrete model

        linear elasticity law with age dependent Youngs modulus modelling the thixotropy
        tensor format!!

    Args:
        mesh : The mesh.
        parameters : Dictionary of material parameters.
        rule: The quadrature rule.
        u: displacement fct
        bc: array of Dirichlet boundaries
        body_force: function of creating body force

    """

    def __init__(
        self,
        mesh: df.mesh.Mesh,
        parameters: dict[str, int | float | str | bool],
        rule: QuadratureRule,
        u: df.fem.Function,
        bc: list[df.fem.DirichletBCMetaClass],
        body_force_fct: Callable,
    ):

        self.p = parameters
        self.rule = rule
        self.mesh = mesh
        dim_to_stress_dim = {1: 1, 2: 4, 3: 9}  # Tensor formulation!
        self.stress_strain_dim = dim_to_stress_dim[self.p["dim"]]

        # generic quadrature function space
        q_V = self.rule.create_quadrature_space(self.mesh)
        q_VT = self.rule.create_quadrature_tensor_space(self.mesh, (self.p["dim"], self.p["dim"]))

        # quadrature functions (required in pde)
        self.q_E = df.fem.Function(q_V, name="youngs_modulus")
        self.q_fd = df.fem.Function(q_V, name="density_increment")

        # path variable from AM Problem
        self.q_array_path = self.rule.create_quadrature_array(self.mesh, shape=1)
        # pseudo density for element activation
        self.q_array_pd = self.rule.create_quadrature_array(self.mesh, shape=1)

        self.q_sig = df.fem.Function(q_VT, name="stress")
        self.q_eps = df.fem.Function(q_VT, name="strain")

        # standard space
        self.V = u.function_space

        # Define variational problem
        v = ufl.TestFunction(self.V)

        # build up form
        # multiplication with activated elements / current Young's modulus
        R_ufl = ufl.inner(self.sigma(u), self.epsilon(v)) * self.rule.dx

        # apply body force
        body_force = body_force_fct(v, self.q_fd, self.rule)
        if body_force:
            R_ufl -= body_force

        # quadrature point part
        self.R = R_ufl

        # derivative
        # normal form
        dR_ufl = ufl.derivative(R_ufl, u)

        # quadrature part
        self.dR = dR_ufl
        self.sigma_evaluator = QuadratureEvaluator(self.sigma(u), self.mesh, self.rule)
        self.eps_evaluator = QuadratureEvaluator(self.epsilon(u), self.mesh, self.rule)

        super().__init__(self.R, u, bc, self.dR)

    def x_sigma(self, v: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """compute stresses for Young's modulus == 1

        Args:
            v: testfunction

        Returns:
            ufl expression for sigma
        """

        x_mu = df.fem.Constant(self.mesh, 1.0 / (2.0 * (1.0 + self.p["nu"])))
        x_lambda = df.fem.Constant(self.mesh, 1.0 * self.p["nu"] / ((1.0 + self.p["nu"]) * (1.0 - 2.0 * self.p["nu"])))
        if self.p["dim"] == 2 and self.p["stress_state"] == "plane_stress":
            # see https://comet-fenics.readthedocs.io/en/latest/demo/elasticity/2D_elasticity.py.html
            x_lambda = df.fem.Constant(self.mesh, 2 * x_mu.value * x_lambda.value / (x_lambda.value + 2 * x_mu.value))

        return 2.0 * x_mu * self.epsilon(v) + x_lambda * ufl.nabla_div(v) * ufl.Identity(self.p["dim"])

    def sigma(self, v: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """computes stresses for real Young's modulus given as quadrature fct q_E

        Args:
            v: testfunction

        Returns:
            ufl expression for sigma
        """

        return self.q_E * self.x_sigma(v)

    def epsilon(self, v: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """computes strains

        Args:
            v: testfunction

        Returns:
            ufl expression for strain
        """
        return ufl.sym(ufl.grad(v))

    def form(self, x: PETSc.Vec) -> None:
        """This function is called before the residual or Jacobian is
        computed. We override it to calculate the values on the quadrature
        functions.
        Args:
           x: The vector containing the latest solution
        """
        self.evaluate_material()
        super().form(x)

    def evaluate_material(self) -> None:
        """evaluate material"""

        # compute current element activation using static function of ConcreteAM
        self.q_array_pd = ConcreteAM.pd_fkt(self.q_array_path)

        # compute current Young's modulus
        # vectorize the function for speed up
        E_fkt_vectorized = np.vectorize(ConcreteAM.E_fkt)
        E_array = E_fkt_vectorized(
            self.q_array_pd,
            self.q_array_path,
            {
                "P0": self.p["E_0"],
                "R_P": self.p["R_E"],
                "A_P": self.p["A_E"],
                "tf_P": self.p["tf_E"],
                "age_0": self.p["age_0"],
            },
        )
        self.q_E.vector.array[:] = E_array
        self.q_E.x.scatter_forward()

        # compute loading factors for density load using static function of ConcreteAM
        fd_array = ConcreteAM.fd_fkt(self.q_array_pd, self.q_array_path, self.p["dt"], self.p["load_time"])
        self.q_fd.vector.array[:] = fd_array
        self.q_fd.x.scatter_forward()

        # postprocessing
        self.sigma_evaluator.evaluate(self.q_sig)
        self.eps_evaluator.evaluate(self.q_eps)  # -> globally in concreteAM not possible why?

    def update_history(self, fields: SolutionFields | None = None, q_fields: QuadratureFields | None = None) -> None:
        """nothing here"""

        pass


# further nonlinear problem classes for different types of materials
# class ConcreteViscoDevThixElasticModel(df.fem.petsc.NonlinearProblem):
