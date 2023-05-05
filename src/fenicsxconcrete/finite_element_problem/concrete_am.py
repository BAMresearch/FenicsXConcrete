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
            "load_time": 1 * ureg("s"),  # load applied in 1 s
        }

        return experiment, model_parameters

    def setup(self) -> None:
        """set up problem"""

        self.rule = QuadratureRule(cell_type=self.mesh.ufl_cell(), degree=self.p["q_degree"])
        self.displacement_space = df.fem.VectorFunctionSpace(self.experiment.mesh, ("CG", self.p["degree"]))

        self.displacement = df.fem.Function(self.displacement_space)
        bcs = self.experiment.create_displacement_boundary(self.displacement_space)
        body_forces = self.experiment.create_body_force(ufl.TestFunction(self.displacement_space))

        # TODO add pseudo density/ add increments

        self.mechanics_problem = self.nonlinear_problem(
            self.experiment.mesh, self.p, self.rule, self.displacement, bcs, body_forces
        )

        # set initial path
        # self.set_initial_path(None)

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

    def solve(self, t: float = 1.0) -> None:
        """define what to do, to solve this problem

        Args:
            t: time point to solve (default = 1)

        """
        #
        self.logger.info(f"solve for t:{t}")

        # how to do incremental stuff here instead of in nonlinear problem?
        self.mechanics_solver.solve(self.displacement)

        # save fields to global problem for sensor output
        self.stress = self.mechanics_problem.q_sig
        self.strain = self.mechanics_problem.q_eps

        # get sensor data
        self.compute_residuals()  # for residual sensor
        for sensor_name in self.sensors:
            # go through all sensors and measure
            self.sensors[sensor_name].measure(self, t)

        # update age & path before next step!
        self.mechanics_problem.update_history()

    def compute_residuals(self) -> None:
        """define what to do, to compute the residuals. Called in solve"""

        self.residual = ufl.action(self.mechanics_problem.R, self.displacement)

    def set_timestep(self, dt: float) -> None:
        self.mechanics_problem.set_timestep(dt)

    # def set_initial_path(self, path: df.fem.Function | None):
    #     """set initial path for problem as (DG, 0) space
    #
    #     Args:
    #         path: function describing the negative time when an element will be reached on space (DG, 0)
    #
    #     """
    #     if path:
    #         # interpolate given path function onto quadrature space
    #
    #         self.mechanics_problem.path.interpolate(path)
    #     else:
    #         # default path
    #         self.mechanics_problem.path = np.zeros_like(self.mechanics_problem.path[:])
    #         self.mechanics_problem.path.x.scatter_forward()

    def pv_plot(self, t=0) -> None:

        xdmf_file = df.io.XDMFFile(self.mesh.comm, self.pv_output_file, "w")
        xdmf_file.write_mesh(self.mesh)
        xdmf_file.write_function(self.displacement, t)

        sigma_plot = project(self.mechanics_problem.sigma(self.displacement), self.plot_space_stress, self.rule.dx)
        E_plot = project(self.mechanics_problem.q_E, self.plot_space, self.rule.dx)

        E_plot.name = "Young's_Modulus"
        sigma_plot.name = "Stress"

        # xdmf_file.write_function(sigma_plot, t)
        # xdmf_file.write_function(E_plot, t)


class ConcreteThixElasticModel(df.fem.petsc.NonlinearProblem):
    """linear elastic thixotropy concrete model

        linear elasticity law with age dependent Youngs modulus modelling the thixotropy
        tensor format!!
        ????incremental formulated u = u_old + du solve for du with given load increment df (using function q_fd) and material evaluation for dsigma/depsilon ???


    Args:
        mesh : The mesh.
        parameters : Dictionary of material parameters.
        rule: The quadrature rule.
        pv_name: Name of the output file.

    """

    def __init__(
        self,
        mesh: df.mesh.Mesh,
        parameters: dict[str, int | float | str | bool],
        rule: QuadratureRule,
        u: df.fem.Function,
        bc: list[df.fem.DirichletBCMetaClass],
        body_force: ufl.form.Form | None,
    ):

        self.p = parameters
        self.rule = rule
        self.mesh = mesh
        dim_to_stress_dim = {1: 1, 2: 4, 3: 9}  # Tensor formulation!
        self.stress_strain_dim = dim_to_stress_dim[self.p["dim"]]

        # generic quadrature function space
        q_V = self.rule.create_quadrature_space(self.mesh)
        q_VT = self.rule.create_quadrature_tensor_space(self.mesh, (self.p["dim"], self.p["dim"]))

        # quadrature functions
        self.q_E = df.fem.Function(q_V, name="youngs_modulus")
        # self.pd = df.fem.Function(q_V, name="pseudo density")

        # path variable fixing to 0
        self.q_array_path = self.rule.create_quadrature_array(self.mesh, shape=1)
        self.q_array_path[:] = 0.0
        # pseudo density
        self.q_array_pd = self.rule.create_quadrature_array(self.mesh, shape=1)

        self.q_sig = df.fem.Function(q_VT, name="stress")
        self.q_eps = df.fem.Function(q_VT, name="strain")
        # self.q_array_sigma = self.rule.create_quadrature_array(self.mesh, shape=self.stress_strain_dim)
        # self.q_array_eps = self.rule.create_quadrature_array(self.mesh, shape=self.stress_strain_dim)

        # standard space
        self.V = u.function_space

        # Define variational problem
        v = ufl.TestFunction(self.V)

        # build up form
        # multiplication with activated elements / current Young's modulus
        R_ufl = ufl.inner(self.sigma(u), self.epsilon(v)) * self.rule.dx

        if body_force:
            R_ufl -= body_force  # TODO where activation?

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

        """
        x_mu = df.fem.Constant(self.mesh, 1.0 / (2.0 * (1.0 + self.p["nu"])))
        x_lambda = df.fem.Constant(self.mesh, 1.0 * self.p["nu"] / ((1.0 + self.p["nu"]) * (1.0 - 2.0 * self.p["nu"])))
        if self.p["dim"] == 2 and self.p["stress_state"] == "plane_stress":
            # see https://comet-fenics.readthedocs.io/en/latest/demo/elasticity/2D_elasticity.py.html
            x_lambda = df.fem.Constant(self.mesh, 2 * x_mu.value * x_lambda.value / (x_lambda.value + 2 * x_mu.value))

        return 2.0 * x_mu * self.epsilon(v) + x_lambda * ufl.nabla_div(v) * ufl.Identity(self.p["dim"])

    def sigma(self, v: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        return self.q_E * self.x_sigma(v)

    def epsilon(self, v: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        return ufl.tensoralgebra.Sym(ufl.grad(v))

    def E_fkt(self, pd: float, path_time: float, parameters: dict) -> float:

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

    def pd_fkt(self, path_time):
        # pseudo density: decide if layer is there (age >=0 active) or not (age < 0 nonactive!)
        # decision based on current path_time value
        # working on list object directly (no vectorizing necessary):
        l_active = np.zeros_like(path_time)  # 0: non-active

        activ_idx = np.where(path_time >= 0 - 1e-5)[0]
        l_active[activ_idx] = 1.0  # active

        return l_active

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
        self.q_E.vector.array[:] = E_array
        self.q_E.x.scatter_forward()

        # postprocessing
        self.sigma_evaluator.evaluate(self.q_sig)
        self.eps_evaluator.evaluate(self.q_eps)

    def update_history(self):

        # no history field currently
        # update path
        self.q_array_path += self.dt * np.ones_like(self.q_array_path)

    def set_timestep(self, dt):
        self.dt = dt
