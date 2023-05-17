from pathlib import Path, PosixPath

import dolfinx as df
import numpy as np
import pint
import ufl
from petsc4py.PETSc import ScalarType

from fenicsxconcrete.experimental_setup.base_experiment import Experiment
from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity
from fenicsxconcrete.gaussian_random_field import Randomfield


class LinearElasticityGRF(LinearElasticity):
    """Material definition for linear elasticity"""

    def __init__(
        self,
        experiment: Experiment,
        parameters: dict[str, pint.Quantity],
        pv_name: str = "pv_output_full",
        pv_path: str = None,
    ) -> None:
        super().__init__(experiment, parameters, pv_name, pv_path)
        # TODO There should be more elegant ways of doing this
        self.pv_file = Path(pv_path) / (pv_name + ".xdmf")

    def setup(self) -> None:
        self.field_function_space = df.fem.FunctionSpace(self.experiment.mesh, ("CG", 1))
        self.lambda_ = df.fem.Function(self.field_function_space)
        self.mu = df.fem.Function(self.field_function_space)

        lame1, lame2 = self.get_lames_constants()
        self.lambda_.vector[:] = lame1
        self.mu.vector[
            :
        ] = lame2  # make this vector as a fenics constant array. Update the lame1 and lame2 in each iteration.

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

    # Random E and nu fields.
    def random_field_generator(
        self,
        field_function_space,
        cov_name,
        mean,
        correlation_length1,
        correlation_length2,
        variance,
        no_eigen_values,
        ktol,
    ):
        random_field = Randomfield(
            field_function_space,
            cov_name,
            mean,
            correlation_length1,
            correlation_length2,
            variance,
            no_eigen_values,
            ktol,
        )
        # random_field.solve_covariance_EVP()
        return random_field

    def parameters_conversion(self, lognormal_mean, lognormal_sigma):
        from math import sqrt

        normal_mean = np.log(lognormal_mean / sqrt(1 + (lognormal_sigma / lognormal_mean) ** 2))
        normal_sigma = np.log(1 + (lognormal_sigma / lognormal_mean) ** 2)
        return normal_mean, normal_sigma

    def get_lames_constants(
        self,
    ):
        # Random E and nu fields.
        E_mean, E_variance = self.parameters_conversion(self.p["E"], self.p["E_std"])  # 3
        Nu_mean, Nu_variance = self.parameters_conversion(self.p["nu"], self.p["nu_std"])  # 0.03

        self.E_randomfield = self.random_field_generator(
            self.field_function_space,
            "squared_exp",
            E_mean,
            self.p["correlation_length_1"],
            self.p["correlation_length_2"],
            E_variance,
            3,
            0.01,
        )
        self.E_randomfield.create_random_field(_type="random", _dist="LN")

        self.nu_randomfield = self.random_field_generator(
            self.field_function_space,
            "squared_exp",
            Nu_mean,
            self.p["correlation_length_1"],
            self.p["correlation_length_2"],
            Nu_variance,
            3,
            0.01,
        )
        self.nu_randomfield.create_random_field(_type="random", _dist="LN")

        lame1 = (self.E_randomfield.field.vector[:] * self.nu_randomfield.field.vector[:]) / (
            (1 + self.nu_randomfield.field.vector[:]) * (1 - 2 * self.nu_randomfield.field.vector[:])
        )
        lame2 = self.E_randomfield.field.vector[:] / (2 * (1 + self.nu_randomfield.field.vector[:]))
        return lame1, lame2

    # TODO move this to sensor definition!?!?!
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

        self.displacement.name = "Displacement"
        self.E_randomfield.field.name = "E_field"
        self.nu_randomfield.field.name = "nu_field"

        with df.io.XDMFFile(self.mesh.comm, self.pv_output_file, "a") as f:
            f.write_function(self.displacement, _t)
            f.write_function(self.E_randomfield.field, _t)
            f.write_function(self.nu_randomfield.field, _t)
