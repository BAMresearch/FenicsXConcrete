import copy
from collections.abc import Callable

import dolfinx as df
import numpy as np
import pint
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from fenicsxconcrete.experimental_setup.am_multiple_layers import AmMultipleLayers
from fenicsxconcrete.experimental_setup.base_experiment import Experiment
from fenicsxconcrete.finite_element_problem.base_material import MaterialProblem
from fenicsxconcrete.helper import Parameters, QuadratureEvaluator, QuadratureRule, project
from fenicsxconcrete.unit_registry import ureg


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
        nonlinear_problem: df.fem.petsc.NonlinearProblem,
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

        # adding default material parameter, will be overridden by outside input
        default_p = Parameters()
        default_p["stress_state"] = "plane_strain" * ureg("")  # default stress state for 2D optional "plane_stress"

        # updating parameters, overriding defaults
        default_p.update(parameters)

        self.nonlinear_problem = nonlinear_problem

        super().__init__(experiment, default_p, pv_name, pv_path)

    @staticmethod
    def parameter_description() -> dict[str, str]:
        """static method returning a description dictionary for required parameters

        Returns:
            description dictionary

        """
        description = {
            "rho": "density of fresh concrete",
            "nu": "Poissons Ratio",
            "E_0": "Youngs Modulus at age=0",
            "R_E": "Reflocculation (first) rate",
            "A_E": "Structuration (second) rate",
            "t_f": "Reflocculation time (switch point)",
            "age_0": "Start age of concrete",
            "degree": "Polynomial degree for the FEM model",
            "q_degree": "Polynomial degree for which the quadrature rule integrates correctly",
            "load_time": "load applied in 1 s",
        }

        return description

    @staticmethod
    def default_parameters() -> tuple[Experiment, dict[str, pint.Quantity]]:
        """Static method that returns a set of default parameters.
        Returns:
            The default experiment instance and the default parameters as a dictionary.

        """

        # default experiment
        experiment = AmMultipleLayers(AmMultipleLayers.default_parameters())

        model_parameters = {
            # Material parameter for concrete model with structural build-up
            "rho": 2070 * ureg("kg/m^3"),  # density of fresh concrete
            "nu": 0.3 * ureg(""),  # Poissons Ratio
            ### default parameters required for thix elastic model
            # Youngs modulus is changing over age (see E_fkt) following the bilinear approach Kruger et al 2019
            # (https://www.sciencedirect.com/science/article/pii/S0950061819317507) with two different rates
            "E_0": 15000 * ureg("Pa"),  # Youngs Modulus at age=0
            "R_E": 15 * ureg("Pa/s"),  # Reflocculation (first) rate
            "A_E": 30 * ureg("Pa/s"),  # Structuration (second) rate
            "t_f": 300 * ureg("s"),  # Reflocculation time (switch point)
            "age_0": 0 * ureg("s"),  # start age of concrete
            # other model parameters
            "degree": 2 * ureg(""),  # polynomial degree
            "q_degree": 2 * ureg(""),  # quadrature rule
            "load_time": 60 * ureg("s"),  # body force load applied in 1 s
        }

        return experiment, model_parameters

    def setup(self) -> None:
        """set up problem"""

        self.rule = QuadratureRule(cell_type=self.experiment.mesh.ufl_cell(), degree=self.p["q_degree"])
        # print("num q", self.rule.number_of_points(mesh=self.experiment.mesh))
        # displacement space (name V required for sensors!)
        self.V = df.fem.VectorFunctionSpace(self.experiment.mesh, ("CG", self.p["degree"]))
        self.strain_stress_space = self.rule.create_quadrature_tensor_space(
            self.experiment.mesh, (self.p["dim"], self.p["dim"])
        )

        # global variables for all AM problems relevant
        # total displacements
        self.displacement = df.fem.Function(self.V)
        # displacement increment
        self.d_disp = df.fem.Function(self.V)

        # set total strain and stress fields
        self.strain = df.fem.Function(self.strain_stress_space, name="strain")
        self.stress = df.fem.Function(self.strain_stress_space, name="stress")

        # boundaries
        bcs = self.experiment.create_displacement_boundary(self.V)
        body_force_fct = self.experiment.create_body_force_am

        self.mechanics_problem = self.nonlinear_problem(
            self.experiment.mesh,
            self.p,
            self.rule,
            self.d_disp,
            bcs,
            body_force_fct,
        )

        # from dolfinx import log
        #
        # log.set_log_level(log.LogLevel.INFO)

        # setting up the solver
        self.mechanics_solver = df.nls.petsc.NewtonSolver(MPI.COMM_WORLD, self.mechanics_problem)
        self.mechanics_solver.convergence_criterion = "incremental"
        self.mechanics_solver.atol = 1e-9
        self.mechanics_solver.rtol = 1e-8
        self.mechanics_solver.report = True

        if self.p["degree"] == 1:
            self.plot_space = df.fem.FunctionSpace(self.mesh, ("DG", 0))
            self.plot_space_stress = df.fem.TensorFunctionSpace(self.mesh, ("DG", 0))
        else:
            self.plot_space = df.fem.FunctionSpace(self.mesh, ("CG", 1))
            self.plot_space_stress = df.fem.TensorFunctionSpace(self.mesh, ("DG", 1))

        # # set up xdmf file with mesh info
        # with df.io.XDMFFile(self.mesh.comm, self.pv_output_file, "w") as f:
        #     f.write_mesh(self.mesh)

    def solve(self, t: pint.Quantity | float = 1.0) -> None:
        """time incremental solving !

        Args:
            t: time point to solve (default = 1) for output

        """
        #
        self.logger.info(f"solve for t:{t}")
        self.logger.info(f"CHECK if external loads are applied as incremental loads e.g. delta_u(t)!!!")

        # solve problem for current time increment
        self.mechanics_solver.solve(self.d_disp)

        # update total displacement
        self.displacement.vector.array[:] += self.d_disp.vector.array[:]
        self.displacement.x.scatter_forward()

        # save fields to global problem for sensor output
        self.stress.vector.array[:] += self.mechanics_problem.q_sig.vector.array[:]
        self.stress.x.scatter_forward()
        self.strain.vector.array[:] += self.mechanics_problem.q_eps.vector.array[:]
        self.strain.x.scatter_forward()

        self.youngsmodulus = self.mechanics_problem.q_E

        # get sensor data
        self.compute_residuals()  # for residual sensor
        for sensor_name in self.sensors:
            # go through all sensors and measure
            self.sensors[sensor_name].measure(self, t)

        # update path & internal variables before next step!
        self.mechanics_problem.update_history()  # if required
        self.update_path()

    def compute_residuals(self) -> None:
        """defines what to do, to compute the residuals. Called in solve for sensors"""

        self.residual = self.mechanics_problem.R

    def set_timestep(self, dt: pint.Quantity) -> None:
        """sets time step value here and in mechanics problems using base units

        Args:
            dt: time step value with unit
        """
        _dt = dt.to_base_units().magnitude
        self.dt = _dt
        self.mechanics_problem.set_timestep(dt.to_base_units().magnitude)

    def update_path(self) -> None:
        """update path for next time increment"""

        # self.q_array_path += self.dt * np.ones_like(self.q_array_path)
        # self.mechanics_problem.q_array_path = self.q_array_path
        self.mechanics_problem.q_array_path += self.dt * np.ones_like(self.mechanics_problem.q_array_path)

    def set_initial_path(self, path: list[float]):
        """set initial path for problem

        Args:
            path: array describing the negative time when an element will be reached on quadrature space

        """
        self.mechanics_problem.q_array_path = path

    def pv_plot(self, t: pint.Quantity | float = 1) -> None:
        """creates paraview output at given time step

        Args:
            t: time point of output (default = 1)
        """
        print("create pv plot for t", t)
        try:
            _t = t.magnitude
        except:
            _t = t

        # write displacement field into existing xdmf file f
        self.displacement.name = "displacement"

        # write further fields
        sigma_plot = project(self.mechanics_problem.sigma(self.displacement), self.plot_space_stress, self.rule.dx)
        E_plot = project(self.mechanics_problem.q_E, self.plot_space, self.rule.dx)

        E_plot.name = "Youngs_Modulus"
        sigma_plot.name = "Stress"

        with df.io.XDMFFile(self.mesh.comm, self.pv_output_file, "a") as f:
            f.write_function(self.displacement, _t)
            f.write_function(sigma_plot, _t)
            f.write_function(E_plot, _t)


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
        body_force:function of cretaing body force

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
        self.q_array_path[:] = 0.0  # default all active set by "set_initial_path(q_array_path)"
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
        return ufl.tensoralgebra.Sym(ufl.grad(v))

    def E_fkt(self, pd: float, path_time: float, parameters: dict) -> float:
        """computes the Young's modulus at current quadrature point according to bilinear Kruger model

        Args:
            pd: value of pseudo density [0 - non active or 1 - active]
            path_time: process time value
            parameters: parameter dict for bilinear model (E_0,R_E,A_E,t_f,age_0)

        Returns:
            value of current Young's modulus
        """
        # print(parameters["age_0"] + path_time)
        if pd > 0:  # element active, compute current Young's modulus
            age = parameters["age_0"] + path_time  # age concrete
            if age < parameters["t_f"]:
                E = parameters["E_0"] + parameters["R_E"] * age
            elif age >= parameters["t_f"]:
                E = (
                    parameters["E_0"]
                    + parameters["R_E"] * parameters["t_f"]
                    + parameters["A_E"] * (age - parameters["t_f"])
                )
        else:
            E = 1e-4  # non-active

        return E

    def pd_fkt(self, path_time: list[float]) -> list[float]:
        """computes pseudo density array

        pseudo density: decides if layer is there (age >=0 active) or not (age < 0 nonactive!)
        decision based on current path_time value

        Args:
            path_time: array of process time values at quadrature points

        Returns:
            array of pseudo density
        """

        l_active = np.zeros_like(path_time)  # 0: non-active

        activ_idx = np.where(path_time >= 0 - 1e-5)[0]
        l_active[activ_idx] = 1.0  # active

        return l_active

    def fd_fkt(self, pd: list[float], path_time: list[float]) -> list[float]:
        """computes weighting fct for body force term in pde

        body force can be applied in several loading steps given by parameter ["load_time"]
        load factor for each step = 1 / "load_time" * dt

        Args:
            pd: array of pseudo density values
            path_time: array of process time values

        Returns:
            array of incremental weigths for body force
        """
        fd = np.zeros_like(pd)

        active_idx = np.where(pd > 0)[0]  # only active elements
        load_idx = np.where(path_time[active_idx] < self.p["load_time"])
        for _ in load_idx:
            fd[active_idx[load_idx]] = self.dt / self.p["load_time"]  # linear ramp

        return fd

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
        print("path time", self.q_array_path)

        # compute current element activation
        self.q_array_pd = self.pd_fkt(self.q_array_path)
        print("pd", self.q_array_pd)

        # defining required parameters as sub dict
        param_E = {}
        param_E["t_f"] = self.p["t_f"]
        param_E["E_0"] = self.p["E_0"]
        param_E["R_E"] = self.p["R_E"]
        param_E["A_E"] = self.p["A_E"]
        param_E["age_0"] = self.p["age_0"]

        # vectorize the function for speed up
        E_fkt_vectorized = np.vectorize(self.E_fkt)
        E_array = E_fkt_vectorized(self.q_array_pd, self.q_array_path, param_E)
        print("E", E_array)
        self.q_E.vector.array[:] = E_array
        self.q_E.x.scatter_forward()

        # compute loading factors for density load
        fd_array = self.fd_fkt(self.q_array_pd, self.q_array_path)
        # print("fd", fd_array)
        self.q_fd.vector.array[:] = fd_array
        self.q_fd.x.scatter_forward()

        # postprocessing
        self.sigma_evaluator.evaluate(self.q_sig)
        self.eps_evaluator.evaluate(self.q_eps)  # -> globally in concreteAM not possible why?

    def update_history(self) -> None:
        """nothing here"""

        pass

    def set_timestep(self, dt: float) -> None:
        """set time step value

        Args:
            dt: value of time step
        """

        self.dt = dt
