import copy
from collections.abc import Callable

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
        nonlinear_problem: df.fem.petsc.NonlinearProblem | None = None,
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

        if nonlinear_problem:
            self.nonlinear_problem = nonlinear_problem
        else:
            self.nonlinear_problem = ConcreteThixElasticModel  # default material

        super().__init__(experiment, default_p, pv_name, pv_path)

    @staticmethod
    def parameter_description() -> dict[str, str]:
        """static method returning a description dictionary for required parameters

        Returns:
            description dictionary

        """
        description = {
            "general parameters": {
                "rho": "density of fresh concrete",
                "nu": "Poissons Ratio",
                "degree": "Polynomial degree for the FEM model",
                "q_degree": "Polynomial degree for which the quadrature rule integrates correctly",
                "load_time": "load applied in 1 s",
                "stress_state": "for 2D plain stress or plane strain",
                "dt": "time step",  # default set in material base class
            },
            "ThixElasticModel": {
                "E_0": "Youngs Modulus at age=0",
                "R_E": "Reflocculation (first) rate",
                "A_E": "Structuration (second) rate",
                "tf_E": "Reflocculation time (switch point)",
                "age_0": "Start age of concrete",
            },
            "ViscoDevThixElasticModel": {
                "visco_case": "which viscoelastic model cmaxwell or ckelvin",
                "age_0": "Start age of concrete",
                "E_0": "linear Youngs Modulus at age=0",
                "R_E": "Reflocculation (first) rate of linear Youngs Modulus",
                "A_E": "Structuration (second) rate of linear Youngs Modulus",
                "tf_E": "Reflocculation time (switch point) of linear Youngs Modulus",
                "E1_0": "visco Youngs Modulus at age=0",
                "R_E1": "Reflocculation (first) rate of visco Youngs Modulus",
                "A_E1": "Structuration (second) rate of visco Youngs Modulus",
                "tf_E1": "Reflocculation time (switch point) of visco Youngs Modulus",
                "eta_0": "damper modulus at age=0",
                "R_eta": "Reflocculation (first) rate of damper modulus",
                "A_eta": "Structuration (second) rate of damper modulus",
                "tf_eta": "Reflocculation time (switch point) of damper modulus",
            },
        }

        return description

    @staticmethod
    def default_parameters(
        non_linear_problem: df.fem.petsc.NonlinearProblem | None = None,
    ) -> tuple[Experiment, dict[str, pint.Quantity]]:
        """Static method that returns a set of default parameters for the selected nonlinear problem.

        Returns:
            The default experiment instance and the default parameters as a dictionary.

        """

        # default experiment
        experiment = AmMultipleLayers(AmMultipleLayers.default_parameters())

        # default parameters according given nonlinear problem
        joined_parameters = {
            # Material parameter for concrete model with structural build-up
            "rho": 2070 * ureg("kg/m^3"),  # density of fresh concrete
            "nu": 0.3 * ureg(""),  # Poissons Ratio
            # other model parameters
            # "degree": 2 * ureg(""),  # polynomial degree --> default defined in base_experiment!!
            "q_degree": 2 * ureg(""),  # quadrature rule
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

        elif non_linear_problem == ConcreteViscoDevThixElasticModel:
            model_parameters = {
                ### default parameters required for ViscoDevThixElasticModel
                "visco_case": "CKelvin" * ureg(""),  # type of viscoelastic model (CKelvin or CMaxwell)
                # Moduli
                "E_0": 70e3 * ureg("Pa"),  # Youngs Modulus at age=0
                "R_E": 0 * ureg("Pa/s"),  # Reflocculation (first) rate for linear Youngs Modulus
                "A_E": 0 * ureg("Pa/s"),  # Structuration (second) rate for linear Youngs Modulus
                "tf_E": 0 * ureg("s"),  # Reflocculation time (switch point) for linear Youngs Modulus
                "E1_0": 20e3 * ureg("Pa"),  # visco Youngs Modulus at age=0
                "R_E1": 0 * ureg("Pa/s"),  # Reflocculation (first) rate for visco Youngs Modulus
                "A_E1": 0 * ureg("Pa/s"),  # Structuration (second) rate for visco Youngs Modulus
                "tf_E1": 0 * ureg("s"),  # Reflocculation time (switch point) for visco Youngs Modulus
                "eta_0": 2e3 * ureg("Pa*s"),  # damper modulus at age=0
                "R_eta": 0 * ureg("Pa"),  # Reflocculation (first) rate for damper modulus
                "A_eta": 0 * ureg("Pa"),  # Structuration (second) rate for damper modulus
                "tf_eta": 0 * ureg("s"),  # Reflocculation time (switch point) for damper modulus
                "age_0": 0 * ureg("s"),  # start age of concrete
            }
        else:
            raise ValueError("non_linear_problem not supported")

        return experiment, {**joined_parameters, **model_parameters}

    def setup(self) -> None:
        """set up problem"""

        self.rule = QuadratureRule(cell_type=self.mesh.ufl_cell(), degree=self.p["q_degree"])
        # print("num q", self.rule.number_of_points(mesh=self.experiment.mesh))
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
            visco_strain=df.fem.Function(self.strain_stress_space, name="visco_strain"),
        )

        # # total displacements
        # self.displacement = df.fem.Function(self.V)
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
        # for visco problem
        if self.nonlinear_problem == ConcreteViscoDevThixElasticModel:
            self.q_fields.visco_strain.vector.array[:] += self.mechanics_problem.q_epsv.vector.array[:]
            self.q_fields.visco_strain.x.scatter_forward()

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

    def set_initial_path(self, path: list[float] | float):
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
        """creates paraview output at given time step

        Args:
            t: time point of output (default = 1)
        """
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
class ConcreteViscoDevThixElasticModel(df.fem.petsc.NonlinearProblem):
    """viscoelastic-thixotropy material model

        derived from 1D Three Parameter Model with age dependent parameters E_0, E_1, eta

        two options: param['visco_case']=='cmaxwell' -> Maxwell chain with n=1! == linear standard solid model (Maxwell in parallel with spring)
                                                       ---------spring(E_0)-------
                                                       |                          |
                                                       --damper(eta)--spring(E_1)--
                     param['visco_case']=='ckelvin' --> Kelvin chain with n=1! == Kelvin plus spring (in Reihe)
                                                          ------spring(E_1)------
                                          ---spring(E_0)--|                     |
                                                          ------damper(eta)------
        with deviatoric assumptions for 3D generalization:
        Deviatoric assumption: vol part of visco strain == 0 damper just working on deviatoric part!
        in tensor format!!

        time integration: BACKWARD EULER


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
        self.q_E1 = df.fem.Function(q_V, name="visco_modulus")
        self.q_eta = df.fem.Function(q_V, name="damper_modulus")
        self.q_fd = df.fem.Function(q_V, name="density_increment")

        # path variable from AM Problem
        self.q_array_path = self.rule.create_quadrature_array(self.mesh, shape=1)
        self.q_array_path[:] = 0.0  # default all active set by "set_initial_path(q_array_path)"
        # pseudo density for element activation
        self.q_array_pd = self.rule.create_quadrature_array(self.mesh, shape=1)

        # stress and strains for viscosity
        self.q_sig = df.fem.Function(q_VT, name="stress")
        self.q_eps = df.fem.Function(q_VT, name="strain")
        self.q_epsv = df.fem.Function(q_VT, name="visco_strain")

        self.q_array_sig1_ten = self.rule.create_quadrature_array(self.mesh, shape=(self.p["dim"], self.p["dim"]))
        self.q_array_sig_old = self.rule.create_quadrature_array(self.mesh, shape=(self.p["dim"], self.p["dim"]))
        self.q_array_epsv_old = self.rule.create_quadrature_array(self.mesh, shape=(self.p["dim"], self.p["dim"]))

        # standard space
        self.V = u.function_space

        # Define variational problem
        self.u = u
        self.u_old = df.fem.Function(self.V)
        v = ufl.TestFunction(self.V)

        # build up form
        # multiplication with activated elements / current Young's modulus
        R_ufl = ufl.inner(self.sigma(self.u), self.epsilon(v)) * self.rule.dx
        R_ufl += -ufl.inner(self.sigma_2(), self.epsilon(v)) * self.rule.dx  # visco part

        # apply body force
        body_force = body_force_fct(v, self.q_fd, self.rule)
        if body_force:
            R_ufl -= body_force

        # quadrature point part
        self.R = R_ufl

        # derivative
        # normal form
        dR_ufl = ufl.derivative(R_ufl, self.u)

        # quadrature part
        self.dR = dR_ufl
        self.sigma_evaluator = QuadratureEvaluator(self.sigma(self.u) - self.sigma_2(), self.mesh, self.rule)
        self.eps_evaluator = QuadratureEvaluator(self.epsilon(self.u), self.mesh, self.rule)
        self.sig1_ten = QuadratureEvaluator(self.sigma_1(self.u_old + self.u), self.mesh, self.rule)  # for visco part

        super().__init__(self.R, self.u, bc, self.dR)

    def sigma(self, v: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """total stress without visco part

        Args:
            v: testfunction

        Returns:
            ufl expression for sigma
        """

        mu_E0 = self.q_E / (2.0 * (1.0 + self.p["nu"]))
        lmb_E0 = self.q_E * self.p["nu"] / ((1.0 + self.p["nu"]) * (1.0 - 2.0 * self.p["nu"]))

        if self.p["dim"] == 2 and self.p["stress_state"] == "plane_stress":
            # see https://comet-fenics.readthedocs.io/en/latest/demo/elasticity/2D_elasticity.py.html
            lmb_E0 = 2 * mu_E0 * lmb_E0 / (lmb_E0 + 2 * mu_E0)
        if self.p["visco_case"].lower() == "cmaxwell":
            # stress stiffness zero + stress stiffness one
            sig = (
                2.0 * mu_E0 * self.epsilon(v)
                + lmb_E0 * ufl.nabla_div(v) * ufl.Identity(self.p["dim"])
                + self.sigma_1(v)
            )
        elif self.p["visco_case"].lower() == "ckelvin":
            # stress stiffness zero
            sig = 2.0 * mu_E0 * self.epsilon(v) + lmb_E0 * ufl.nabla_div(v) * ufl.Identity(self.p["dim"])
        else:
            sig = None
            raise ValueError("case not defined")

        return sig

    def sigma_1(self, v: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """stress for visco strain computation

        Args:
            v: testfunction

        Returns:
            ufl expression for sigma
        """

        if self.p["visco_case"].lower() == "cmaxwell":
            mu_E1 = self.q_E1 / (2.0 * (1.0 + self.p["nu"]))
            lmb_E1 = self.q_E1 * self.p["nu"] / ((1.0 + self.p["nu"]) * (1.0 - 2.0 * self.p["nu"]))
            if self.p["dim"] == 2 and self.p["stress_state"] == "plane_stress":
                lmb_E1 = 2 * mu_E1 * lmb_E1 / (lmb_E1 + 2 * mu_E1)

            sig1 = 2.0 * mu_E1 * self.epsilon(v) + lmb_E1 * ufl.nabla_div(v) * ufl.Identity(self.p["dim"])
        elif self.p["visco_case"].lower() == "ckelvin":
            sig1 = self.sigma(v)
        else:
            sig1 = None
            raise ValueError("case not defined")

        return sig1

    def sigma_2(self) -> ufl.core.expr.Expr:
        """damper stress related to epsv

        Args:
            v: testfunction

        Returns:
            ufl expression for sigma
        """

        if self.p["visco_case"].lower() == "cmaxwell":
            mu_E1 = self.q_E1 / (2.0 * (1.0 + self.p["nu"]))
            sig2 = 2 * mu_E1 * self.q_epsv
        elif self.p["visco_case"].lower() == "ckelvin":
            mu_E0 = self.q_E / (2.0 * (1.0 + self.p["nu"]))
            sig2 = 2 * mu_E0 * self.q_epsv
        else:
            sig2 = None
            raise ValueError("case not defined")
        return sig2

    def epsilon(self, v: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """computes strains

        Args:
            v: testfunction

        Returns:
            ufl expression for strain
        """
        return ufl.tensoralgebra.Sym(ufl.grad(v))

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

        # compute current element activation using static function from ConcreteAM
        self.q_array_pd = ConcreteAM.pd_fkt(self.q_array_path)
        print("current path", self.q_array_path)

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
        print("E_array", E_array)
        E1_array = E_fkt_vectorized(
            self.q_array_pd,
            self.q_array_path,
            {
                "P0": self.p["E1_0"],
                "R_P": self.p["R_E1"],
                "A_P": self.p["A_E1"],
                "tf_P": self.p["tf_E1"],
                "age_0": self.p["age_0"],
            },
        )
        self.q_E1.vector.array[:] = E1_array
        self.q_E1.x.scatter_forward()
        print("E1_array", E1_array)
        Eta_array = E_fkt_vectorized(
            self.q_array_pd,
            self.q_array_path,
            {
                "P0": self.p["eta_0"],
                "R_P": self.p["R_eta"],
                "A_P": self.p["A_eta"],
                "tf_P": self.p["tf_eta"],
                "age_0": self.p["age_0"],
            },
        )
        self.q_eta.vector.array[:] = Eta_array
        self.q_eta.x.scatter_forward()
        print("Eta_array", Eta_array)
        # compute loading factors for density load using static function from ConcreteAM
        fd_array = ConcreteAM.fd_fkt(self.q_array_pd, self.q_array_path, self.p["dt"], self.p["load_time"])
        self.q_fd.vector.array[:] = fd_array
        self.q_fd.x.scatter_forward()
        print("fd_array", fd_array)

        # compute current visco strains and stresses
        print("delta_u", self.u.vector.array[:])
        self.sigma_evaluator.evaluate(self.q_sig)
        self.eps_evaluator.evaluate(self.q_eps)  # -> globally in concreteAM not possible why?
        print("sig", self.q_sig.vector.array[:])
        print("eps", self.q_eps.vector.array[:])
        print("epsv", self.q_epsv.vector.array[:])

        # compute delta visco strain
        print("u_old", self.u_old.vector.array[:])
        self.sig1_ten.evaluate(self.q_array_sig1_ten)
        input()

        if self.p["visco_case"].lower() == "cmaxwell":
            print("in cmaxwell")
            # list of mu at each quadrature point [mu independent of plane stress or plane strain]
            mu_E1 = 0.5 * E1_array / (1.0 + self.p["nu"])
            # factor at each quadrature point
            factor = 1 + self.p["dt"] * 2.0 * mu_E1 / Eta_array
            # repeat material parameters to size of epsv and compute epsv
            # & reshaped material parameters per eps entry!!
            self.new_epsv = (
                1.0
                / np.repeat(factor, self.p["dim"] ** 2)
                * (
                    self.q_array_epsv_old
                    + self.p["dt"] / np.repeat(Eta_array, self.p["dim"] ** 2) * self.q_array_sig1_ten
                )
            )
            # compute delta visco strain
            self.q_epsv.vector.array[:] = self.new_epsv - self.q_array_epsv_old

        elif self.p["visco_case"].lower() == "ckelvin":
            print("in ckelvin")
            # list of mu_1 and mu_0 at each quadrature point
            mu_E1 = 0.5 * E1_array / (1.0 + self.p["nu"])
            mu_E0 = 0.5 * E_array / (1.0 + self.p["nu"])
            # factor at each quadrature point
            factor = 1 + self.p["dt"] * 2.0 * mu_E0 / Eta_array + self.p["dt"] * 2.0 * mu_E1 / Eta_array
            # repeat material parameters to size of epsv and compute epsv
            self.new_epsv = (
                1.0
                / np.repeat(factor, self.p["dim"] ** 2)
                * (
                    self.q_array_epsv_old
                    + self.p["dt"] / np.repeat(Eta_array, self.p["dim"] ** 2) * self.q_array_sig1_ten
                )
            )
            # compute delta visco strain
            self.q_epsv.vector.array[:] = self.new_epsv - self.q_array_epsv_old

        else:
            raise ValueError("visco case not defined")

    def update_history(self, fields: SolutionFields | None = None, q_fields: QuadratureFields | None = None) -> None:
        """set array values for old time using current solution"""
        self.q_array_epsv_old[:] = q_fields.visco_strain.vector.array[:]

        self.q_array_sig_old[:] = q_fields.stress.vector.array[:]

        self.u_old.vector.array[:] = fields.displacement.vector.array[:]
        self.u_old.x.scatter_forward()
