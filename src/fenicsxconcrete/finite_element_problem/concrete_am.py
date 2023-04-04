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
from fenicsxconcrete.helper import Parameters, QuadratureEvaluator, QuadratureRule
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
        nonlinear_problem: df.fem.petsc.NonlinearProblem,
        pv_name: str = "pv_output_full",
        pv_path: str | None = None,
    ) -> None:

        # adding default material parameter, will be overridden by outside input
        default_p = Parameters()

        # updating parameters, overriding defaults
        default_p.update(parameters)

        self.nonlinear_problem = nonlinear_problem

        super().__init__(experiment, default_p, pv_name, pv_path)

    @staticmethod
    def parameter_description() -> dict[str, str]:
        description = {
            "rho": "density of fresh concrete",
            "nu": "Poissons Ratio",
            "E_0": "Youngs Modulus at age=0",
            "R_E": "Reflocculation (first) rate",
            "A_E": "Structuration (second) rate",
            "t_f": "Reflocculation time (switch point)",
            "age_0": "Start age of concrete",
            "degree": "Polynomial degree for the FEM model",
            "u_bc": "Displacement on top for boundary conditions type",  # TODO
            "load_time": "load applied in 1 s",
        }

        return description

    @staticmethod
    def default_parameters() -> tuple[Experiment, dict[str, pint.Quantity]]:
        """
        Static method that returns a set of default parameters.
        Returns:
            The default parameters as a dictionary.
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
            "u_bc": 0.1 * ureg(""),  # displacement on top
            "load_time": 1 * ureg("s"),  # load applied in 1 s
        }

        return experiment, model_parameters

    def setup(self) -> None:
        # set up problem

        self.displacement = df.fem.VectorFunctionSpace(self.experiment.mesh, ("CG", self.p["degree"]))
        self.pseudo_density = df.fem.VectorFunctionSpace(self.experiment.mesh, ("DG", 0))

        bcs = self.experiment.create_displacement_boundary(self.displacement)
        external_forces = self.experiment.create_force_boundary(ufl.TestFunction(self.displacement),pd=self.pseudo_density)
        body_forces = self.experiment.create_body_force(ufl.TestFunction(self.displacement),pd=self.pseudo_density)


        try:
            self.mechanics_problem = self.nonlinear_problem(self.experiment.mesh,
                                                            self.p,
                                                            self.displacement,
                                                            bcs,
                                                            body_forces,
                                                            external_forces)
        except:
            raise ValueError(f"nonlinear problem {self.nonlinear_problem} not yet implemented")


        # setting up the solver
        self.mechanics_solver = df.nls.petsc.NewtonSolver(MPI.COMM_WORLD, self.mechanics_problem)
        self.mechanics_solver.convergence_criterion = "incremental"
        self.mechanics_solver.atol = 1e-9
        self.mechanics_solver.rtol = 1e-8
        self.mechanics_solver.report = True

        # self.V = self.mechanics_problem.V  # for reaction force sensor
        # self.residual = None  # initialize

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
        self.mechanics_problem.update_history()

    def compute_residuals(self) -> None:
        # define what to do, to compute the residuals. Called in solve
        self.residual = ufl.action(self.mechanics_problem.R, self.displacement)

    def set_initial_path(self, path):
        self.mechanics_problem.set_initial_path(path)


class ConcreteThixElasticModel(df.fem.petsc.NonlinearProblem):
    """ incremental linear elastic thixotropy concrete model

        linear elasticity law with age dependent Youngs modulus modelling the thixotropy
        tensor format!!
        incremental formulated u = u_old + du solve for du with given load increment df (using function q_fd) and material evaluation for dsigma/depsilon


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
        bcs: list[df.fem.DirichletBCMetaClass],
        body_forces: ufl.form.Form,
    ):
        self.p_magnitude = parameters
        dim_to_stress_dim = {1: 1, 2: 3, 3: 6}
        self.stress_strain_dim = dim_to_stress_dim[self.p_magnitude["dim"]]
        self.rule = rule

        if self.p_magnitude["degree"] == 1:
            self.visu_space = df.fem.FunctionSpace(mesh, ("DG", 0))
            self.visu_space_T = df.fem.TensorFunctionSpace(mesh, ("DG", 0))
        else:
            self.visu_space = df.fem.FunctionSpace(mesh, ("CG", 1))
            self.visu_space_T = df.fem.TensorFunctionSpace(mesh, ("CG", 1))

        # interface to problem for sensor output: # here tensor format is used for e_eps/q_sig
        self.visu_space_eps = self.visu_space_T
        self.visu_space_sig = self.visu_space_T

        # standard space
        self.V = df.fem.VectorFunctionSpace(mesh, ("CG", self.p_magnitude["degree"]))
        # space for element activation: constant per element
        self.V_pd = df.fem.VectorFunctionSpace(mesh, ("DG", 0))

        # generic quadrature function space
        q_V = self.rule.create_quadrature_space(mesh)
        q_VT = self.rule.create_quadrature_vector_space(mesh, dim=self.stress_strain_dim)

        # quadrature functions
        self.q_path = df.Function(self.V_pd, name="path time")
        self.q_pd = df.Function(self.V_pd, name="pseudo density")

        self.q_E = df.fem.Function(q_V, name="youngs modulus")

        # self.q_fd = df.fem.Function(q_V, name="load factor")  # for density ???

        self.q_eps = df.fem.Function(q_VT, name="strain")
        self.q_sig = df.fem.Function(q_VT, name="stress")
        self.q_dsig = df.fem.Function(q_VT, name="delta stress")
        self.q_sig_old = df.fem.Function(q_VT, name="old stress")

        self.u_old = df.fem.Function(self.V, name="old displacement")
        self.u = df.fem.Function(self.V, name="displacement")

        # Define variational problem
        self.du = df.Function(self.V,  name="delta displacements")
        v = df.TestFunction(self.V)

        # build up form
        # multiplication with activated elements / current Young's modulus
        R_ufl = self.q_E * df.inner(self.x_sigma(self.du), self.eps(v)) * self.rule.dx

        external_force = self.set_force()
        if self.set_body_force():
            R_ufl -= external_force

        body_force = self.set_body_force()
        if body_force:
            R_ufl -= self.body_force

        # quadrature point part
        self.R = R_ufl

        # derivative
        # normal form
        dR_ufl = df.derivative(R_ufl, self.u)
        # quadrature part
        self.dR = dR_ufl

        self.sigma_evaluator = QuadratureEvaluator(self.sigma_voigt(self.sigma_ufl), mesh, self.rule)

    def set_bc(self):
        # ???
    def set_force(self):
        # ???
    def set_body_force():
        #???
    def F(self, x: PETSc.Vec, b: PETSc.Vec):
        super().F(x, b)

    def J(self, x: PETSc.Vec, A: PETSc.Mat):
        super().J(x, A)
