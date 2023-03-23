import dolfinx as df
import numpy as np
import pint
import ufl
from petsc4py.PETSc import ScalarType

from fenicsxconcrete.experimental_setup.base_experiment import Experiment
from fenicsxconcrete.experimental_setup.cantilever_beam import CantileverBeam
from fenicsxconcrete.finite_element_problem.base_material import MaterialProblem
from fenicsxconcrete.helper import Parameters
from fenicsxconcrete.unit_registry import ureg


class LinearElasticity(MaterialProblem):
    """Material definition for linear elasticity"""

    def __init__(
        self,
        experiment: Experiment,
        parameters: dict[str, pint.Quantity],
        pv_name: str = "pv_output_full",
        pv_path: str = None,
    ) -> None:
        """defines default parameters, for the rest, see base class"""

        # adding default material parameter, will be overridden by outside input
        default_p = Parameters()
        # default_p['dummy'] = 'example' * ureg('')  # example default parameter for this class

        # updating parameters, overriding defaults
        default_p.update(parameters)

        super().__init__(experiment, default_p, pv_name, pv_path)

    def setup(self) -> None:
        # compute different set of elastic moduli
        self.lambda_ = df.fem.Constant(
            self.mesh,
            self.p["E"] * self.p["nu"] / ((1 + self.p["nu"]) * (1 - 2 * self.p["nu"])),
        )
        self.mu = df.fem.Constant(self.mesh, self.p["E"] / (2 * (1 + self.p["nu"])))

        # define function space ets.
        self.V = df.fem.VectorFunctionSpace(self.mesh, ("Lagrange", self.p["degree"]))  # 2 for quadratic elements
        self.V_scalar = df.fem.FunctionSpace(self.mesh, ("Lagrange", self.p["degree"]))

        # Define variational problem
        self.u_trial = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)

        # initialize L field, not sure if this is the best way...
        zero_field = df.fem.Constant(self.mesh, ScalarType(np.zeros(self.p["dim"])))
        self.L = ufl.dot(zero_field, self.v) * ufl.dx

        # apply external loads
        external_force = self.experiment.create_force_boundary(self.v)
        if external_force:
            self.L = self.L + external_force

        body_force = self.experiment.create_body_force(self.v)
        if body_force:
            self.L = self.L + body_force

        # boundary conditions only after function space
        bcs = self.experiment.create_displacement_boundary(self.V)

        self.a = ufl.inner(self.sigma(self.u_trial), self.epsilon(self.v)) * ufl.dx
        self.weak_form_problem = df.fem.petsc.LinearProblem(
            self.a,
            self.L,
            bcs=bcs,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )

    @staticmethod
    def default_parameters() -> tuple[Experiment, dict[str, pint.Quantity]]:
        """returns a dictionary with required parameters and a set of working values as example"""
        # default setup for this material
        experiment = CantileverBeam(CantileverBeam.default_parameters())

        model_parameters = {}
        model_parameters["rho"] = 7750 * ureg("kg/m^3")
        model_parameters["E"] = 210e9 * ureg("N/m^2")
        model_parameters["nu"] = 0.28 * ureg("")

        return experiment, model_parameters

    # Stress computation for linear elastic problem
    def epsilon(self, u: ufl.argument.Argument) -> ufl.tensoralgebra.Sym:
        return ufl.tensoralgebra.Sym(ufl.grad(u))

    def sigma(self, u: ufl.argument.Argument) -> ufl.core.expr.Expr:
        return self.lambda_ * ufl.nabla_div(u) * ufl.Identity(self.p["dim"]) + 2 * self.mu * self.epsilon(u)

    def solve(self, t: float = 1.0) -> None:
        self.displacement = self.weak_form_problem.solve()

        # TODO Defined as abstractmethod. Should it depend on sensor instead of material?
        self.compute_residuals()

        # get sensor data
        for sensor_name in self.sensors:
            # go through all sensors and measure
            self.sensors[sensor_name].measure(self, t)

    def compute_residuals(self) -> None:
        self.residual = ufl.action(self.a, self.displacement) - self.L

    # paraview output
    # TODO move this to sensor definition!?!?!
    def pv_plot(self, t: int = 0) -> None:
        # TODO add possibility for multiple time steps???
        # Displacement Plot

        # "Displacement.xdmf"
        # pv_output_file
        with df.io.XDMFFile(self.mesh.comm, self.pv_output_file, "w") as xdmf:
            xdmf.write_mesh(self.mesh)
            xdmf.write_function(self.displacement)
