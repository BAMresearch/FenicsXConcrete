import basix
import dolfinx as df
import numpy as np
import pint
import ufl
from dolfinx.nls.petsc import NewtonSolver
from fenics_constitutive.models import IncrSmallStrainModel, StressStrainConstraint
from fenics_constitutive.solver import CorotationalIncrSmallStrainProblem, ufl_mandel_strain
from mpi4py import MPI

from fenicsxconcrete.experimental_setup import AmMultipleLayers, Experiment
from fenicsxconcrete.finite_element_problem.base_material import MaterialProblem, QuadratureFields, SolutionFields
from fenicsxconcrete.util import QuadratureRule, project, ureg

from fenicsxconcrete.finite_element_problem.material_for_am_fc import LinearElasticityModel


class ConcreteAMFC(MaterialProblem):
    """A class for additive manufacturing models

    - including pseudo density approach for element activation -> set_initial_path == negative time when element will be activated
    - time incremental weak form (in case of density load increments are computed automatic, otherwise user controlled)
    - the FEM solver itself is fenics-constitutive's CorotationalIncrSmallStrainProblem
      (objective/Jaumann stress rate + updated-Lagrangian mesh update); this class only adds the
      additive-manufacturing layer on top of it (element activation, time-dependent material
      parameters and AM-specific output fields)
    - material laws from fenics-constitutive (incremental small strain models)

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
            material: material law IncSmallStrainModel from fenics-constitutive
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
    def default_parameters(
        material: str | None = None,
    ) -> tuple[Experiment, dict[str, pint.Quantity]]:
        """Static method that returns a set of default parameters for the selected nonlinear problem.

        Args:
            material: string of material law as IncSmallStrainModel from fenics-constitutive

        Returns:
            The default experiment instance and the default parameters as a dictionary.

        """

        # default experiment
        experiment = AmMultipleLayers(AmMultipleLayers.default_parameters())

        # default parameters according given nonlinear problem
        parameters = {
            # general parameters
            "rho": 2070 * ureg("kg/m^3"),  # density
            "g": 9.81 * ureg("m/s^2"),  # gravity
            # general model parameters
            "degree": 2 * ureg(""),  # polynomial degree
            "q_degree": 2 * ureg(""),  # quadrature rule
            "dt": 1.0 * ureg("s"),  # time step
            "load_time": 60 * ureg("s"),  # body force load applied in s
            # material parameters
            # ... - according to chosen material law!
        }
        if not material or material == "LinearElasticityModel": # default material
            model_parameters = {
                "E": 15000 * ureg("Pa"),  # Youngs Modulus
                "nu": 0.3 * ureg(""),  # Poisson ratio
                "A_E": 1500 * ureg("Pa/s"),  # rate of change over time
                "time_fct": "linear" * ureg(""),  # time dependency of material parameters
            }
        elif material == "VonMises3D":
            model_parameters = {
                "p_ka": 175000 * ureg("MPa"),  # bulk modulus
                "p_mu": 80769 * ureg("MPa"),  # shear modulus
                "p_y0": 1200 * ureg("MPa"),  # initial yield stress
                "p_y00": 2500 * ureg("MPa"),  # final yield stress
                "p_w": 200 * ureg(""),  # saturation parameter
            }

        elif material == "mohr_coulomb_smoothed_3D_analytical":
            model_parameters = {
                "E": 78000 * ureg("Pa"),
                "nu": 0.3 * ureg(""),  # poisson ratio
                "c_0": 1050 * ureg("Pa"),
                "c_00": 2280 * ureg("Pa"),
                "p_w": 10 * ureg(""),
                "psi": 20 * np.pi / 180 * ureg(""),
                "phi": 20 * np.pi / 180 * ureg(""),
                "theta_T": 26 * np.pi / 180 * ureg(""),
                "a": 0.25 * 1.05 / np.tan(20) * ureg("")
            }

        else:
            raise ValueError("material law not known")

        return experiment, {**parameters, **model_parameters}

    @staticmethod
    def default_material() -> IncrSmallStrainModel:
        """Static method that returns the default material model for the selected nonlinear problem.

        Returns:
            The default material class.

        """

        # default material
        material = LinearElasticityModel

        return  material

    def setup(self) -> None:
        """set up problem"""

        # displacement space and field
        dim = self.experiment.mesh.topology.dim
        self.V = df.fem.functionspace(self.experiment.mesh, ("CG", self.p["degree"], (dim,)))
        self.fields = SolutionFields(displacement=df.fem.Function(self.V, name="displacement"))

        # material law based on the fenics-constitutive interface
        try:
            law = self.material_law(self.p, constraint=StressStrainConstraint.FULL)
        except TypeError:
            law = self.material_law(self.p)
            assert law.constraint == StressStrainConstraint.FULL
        # keep a direct handle on the model so the AM time-dependent parameter
        # updates mutate the very object the solver evaluates
        self.law = law

        # quadrature rule and scalar quadrature space for AM activation / output fields
        # (full-mesh quadrature ordering; for a homogeneous single-law domain the
        # fenics-constitutive solver uses an identity submesh map, so this ordering
        # matches the arrays the constitutive law receives)
        self.rule = QuadratureRule(cell_type=self.mesh.ufl_cell(), degree=self.p["q_degree"])
        QSe = basix.ufl.quadrature_element(
            self.mesh.topology.cell_name(), value_shape=(), degree=self.p["q_degree"]
        )
        s_space = df.fem.functionspace(self.mesh, QSe)
        # pseudo-density for element activation (also scales the body force over time)
        self.density_time = df.fem.Function(s_space, name="density")
        # stored material modulus per gauss point (for sensors / output)
        self.modulus = df.fem.Function(s_space, name="modulus")
        # von Mises stress per gauss point (AM postprocessing output)
        self.von_mises = df.fem.Function(s_space, name="von_mises")
        # path time per quadrature point (negative => element not yet activated)
        self.q_array_path_time = np.zeros_like(self.density_time.x.array[:])

        # boundaries
        bcs = self.experiment.create_displacement_boundary(self.V)

        # AM body force with element activation, injected as an external force; the
        # base IncrSmallStrainProblem subtracts it from the residual. It references
        # self.density_time, so updating that field each step updates the load.
        v = ufl.TestFunction(self.V)
        body_force_form = self.experiment.create_body_force_am(v, self.density_time, self.rule)
        external_forces = [body_force_form] if body_force_form else None

        # the actual FEM solve loop lives in fenics-constitutive: the corotational
        # incremental small-strain problem provides the objective (Jaumann) stress
        # rate and the updated-Lagrangian mesh update the AM print needs.
        self.mechanics_problem = CorotationalIncrSmallStrainProblem(
            law,
            self.fields.displacement,
            bcs,
            self.p["q_degree"],
            del_t=self.p["dt"],
            external_forces=external_forces,
        )
        self.mechanics_problem._time = self.p["dt"]

        # residual form for the reaction-force sensor: fenics-constitutive no longer
        # exposes the assembled residual, so rebuild the same form the solver uses
        # internally (stress test-function pairing minus the external forces).
        dxm = ufl.dx(metadata={"quadrature_degree": self.p["q_degree"], "quadrature_scheme": "default"})
        self._residual_form = (
            ufl.inner(ufl_mandel_strain(v, StressStrainConstraint.FULL), self.mechanics_problem.stress_1) * dxm
        )
        if external_forces:
            self._residual_form -= sum(external_forces)

        # additional output fields
        history_1 = self.mechanics_problem._history_1[0]  # history dict for the single law, or None
        self.a_plot = bool(history_1) and "alpha" in history_1
        q_fields_kwargs = dict(
            measure=self.rule.dx,
            plot_space_type=("CG", 1),
            mandel_stress=self.mechanics_problem.stress_1,
        )
        if self.a_plot:
            q_fields_kwargs["history_scalar"] = history_1["alpha"]
        self.q_fields = QuadratureFields(**q_fields_kwargs)

        self.mandel_stress_dim = law.stress_strain_dim  # for sensor
        self.hist_a = 1

        # setting up the solver
        self.mechanics_solver = NewtonSolver(MPI.COMM_WORLD, self.mechanics_problem)
        self.mechanics_solver.atol = 1e-9
        self.mechanics_solver.rtol = 1e-8
        self.mechanics_solver.report = True

        # for paraview stress output (vector space)
        self.plot_space_stress = df.fem.functionspace(
            self.experiment.mesh,
            (self.q_fields.plot_space_type[0], self.q_fields.plot_space_type[1], (self.mandel_stress_dim,)),
        )
        # history alpha space
        self.plot_space_alpha = df.fem.functionspace(self.experiment.mesh, self.q_fields.plot_space_type)

    def solve(self) -> None:
        """time incremental solving !"""

        self.update_time()  # set t+dt # TODO check with nonlinear problem.time

        self.logger.info(f"solve for t: {self.time}")
        self.logger.info("CHECK if external loads are applied as incremental loads e.g. delta_u(t)!!!")

        # update path and incr loading
        self.update_path()

        # compute current material parameters according to path_time
        self.update_material_parameters()

        # solve problem for current time increment
        n, converged = self.mechanics_solver.solve(self.fields.displacement)
        if not converged:
            self.logger.warning("Mechanics solve did not converge")
        else:
            self.logger.info(f"Mechanics solve converged in {n} iterations")

        # AM-specific output: von Mises stress (the fc solver does not compute it)
        self.von_mises.x.array[:] = self.von_mises_from_mandel(self.mechanics_problem.stress_1.x.array)
        self.von_mises.x.scatter_forward()

        self.mechanics_problem.update()

        # get sensor data
        self.compute_residuals()  # for residual sensor
        for sensor_name in self.sensors:
            # go through all sensors and measure
            self.sensors[sensor_name].measure(self)

    def compute_residuals(self) -> None:
        """defines what to do, to compute the residuals. Called in solve for sensors"""

        self.residual = self._residual_form

    def update_material_parameters(self) -> None:
        """update material parameters at each quadrature point according time based on path_time"""

        # compute material parameters for time t
        if self.material_law.__name__ == "LinearElasticityModel":
            time_params = ["E"]
            p_values = self.get_params_gp(time_params)

            # in the linear model we adapt the factor of the youngs modulus
            self.law.factor = p_values["E"] / self.p["E"]

            # store E factor just for access since material law dependent do it here
            self.modulus.x.array[:] = self.law.factor
            self.modulus.x.scatter_forward()

        elif self.material_law.__name__ == "VonMises3D":
            # changing parameters
            time_params = ["p_ka", "p_mu", "p_y0", "p_y00", "p_w"]
            p_values = self.get_params_gp(time_params)
            #
            self.law.p_ka = p_values["p_ka"]
            self.law.p_mu = p_values["p_mu"]
            self.law.p_y0 = p_values["p_y0"]
            self.law.p_y00 = p_values["p_y00"]
            self.law.p_w = p_values["p_w"]

            # store bulk modulus just for access since material law dependent do it here
            self.modulus.x.array[:] = self.law.p_ka
            self.modulus.x.scatter_forward()

        else:
            raise ValueError("material law not known")

    def get_params_gp(self, time_params):
        """evaluate for a given string list of parameters the current values at each quadrature point
        Args:
            time_params: list of strings with parameter names
        Returns:
            dict with parameter values at each quadrature point
        """

        params = {}
        p_values = {}
        for pi in time_params:
            # get params
            params["P0"] = self.p[pi]
            try:
                params["A_P"] = self.p[f"A_{pi}"]
            except KeyError:
                params["A_P"] = 0.0  # no change
            try:
                params["R_P"], params["tf_P"] = self.p[f"R_{pi}"], self.p[f"tf_{pi}"]
            except KeyError:
                params["R_P"], params["tf_P"] = 0.0, 0.0

            # compute for each quadrature point
            fkt_vectorized = np.vectorize(self.param_time_fkt)
            p_values[pi] = fkt_vectorized(self.q_array_path_time, params, _model=self.p["time_fct"])
        return p_values

    def update_path(self) -> None:
        """update path for next time increment
            and compute density field for element activation including load stepping

        density includes load stepping - active is between 0 and 1
        """

        self.q_array_path_time += self.p["dt"] * np.ones_like(self.q_array_path_time)  # path time

        # compute density field for element activation
        density = np.zeros_like(self.q_array_path_time)  # 0: non-active

        active_idx = np.where(self.q_array_path_time >= 0 - 1e-5)[0]  # only active elements
        density[active_idx] = 1.0  # 1: active

        # load stepping: linear ramp of body force over time of active elements
        load_idx = np.where(self.q_array_path_time[active_idx] <= self.p["load_time"])
        for _ in load_idx:
            density[active_idx[load_idx]] = (
                self.q_array_path_time[active_idx[load_idx]] / self.p["load_time"]
            )  # linear ramp

        self.density_time.x.array[:] = density
        self.density_time.x.scatter_forward()

    def set_initial_path(self, path: list[float] | float) -> None:
        """set initial path for problem

        Args:
            path: array describing the negative time when an element will be reached on quadrature space
                    if only one value is given, it is assumed that all elements are reached at the same time

        """
        if isinstance(path, float):
            self.q_array_path_time = path * np.ones_like(self.q_array_path_time)
        else:
            self.q_array_path_time = path

    def pv_plot(self) -> None:
        """creates paraview output at given time step"""

        self.logger.info(f"create pv plot for t: {self.time}")

        if self.p["degree"] > 1:
            # project displacement to linear space for writing
            V_project = df.fem.functionspace(self.experiment.mesh, ("CG", 1, (self.p["dim"],)))
            disp_plot = df.fem.Function(V_project, name="displacement")
            #disp_plot.interpolate(self.fields.displacement)
            project(self.fields.displacement, V_project, ufl.dx, disp_plot)
            disp_plot.x.scatter_forward()
        else:
            disp_plot = self.fields.displacement
            disp_plot.x.scatter_forward()


        # write further fields
        sigma_plot = project(self.q_fields.mandel_stress, self.plot_space_stress, self.rule.dx)
        sigma_plot.name = "Stress"

        density_plot = project(self.density_time, self.plot_space_alpha, self.rule.dx)
        density_plot.name = "Density"

        Q0 = df.fem.functionspace(self.mesh, ("DG", 0))
        VM_plot = project(self.von_mises, Q0, self.rule.dx)
        VM_plot.name = "Von-Mises_DG0"

        if self.a_plot:
            # the scalar hardening history alpha is stored as a length-1 vector
            # quadrature field (history_dim {"alpha": 1}); take its component so it
            # projects onto the scalar plot space
            alpha = self.q_fields.history_scalar
            alpha_expr = alpha[0] if alpha.ufl_shape == (1,) else alpha
            alpha_plot = project(alpha_expr, self.plot_space_alpha, self.rule.dx)
            alpha_plot.name = "Alpha"
        # #
        ## write to file
        with df.io.XDMFFile(self.mesh.comm, self.pv_output_file, "a") as f:
             f.write_function(disp_plot, self.time)
             f.write_function(sigma_plot, self.time)
             f.write_function(density_plot, self.time)
             f.write_function(VM_plot, self.time)
             if self.a_plot:
                 f.write_function(alpha_plot, self.time)

    @staticmethod
    def param_time_fkt(time, parameters: dict, _model: str = "linear") -> float:
        """computes

        Args:
            time: time value
            parameters: required parameter dict see models
            _model: model type:
                        bilinear model
                                    P(t) = P0 + R_P*t for t< tf_P and P(t) = P0 + R_P*tf_P + A_P * (t-tf_P)
                                    requires parameters (P0: start value t=0, R_P: first rate until tf_P, A_P: second rate after tf_P, tf_P: switch time for rates)
                        linear model
                                    P(t)= P0 + A_P*t
                                    requires parameters (P0: start value (t=0), A_P: rate)
        Returns:
            current parameter value
        """

        value = None
        if _model == "linear":
            if time >= -1e-5:  # only for activated elements
                value = parameters["P0"] + parameters["A_P"] * time
            else:
                value = 1e-4  # for non-active elements
        elif _model == "bilinear":
            if time >= -1e-5 and time < parameters["tf_P"]:
                value = parameters["P0"] + parameters["R_P"] * time
            elif time >= parameters["tf_P"]:
                value = (
                    parameters["P0"]
                    + parameters["R_P"] * parameters["tf_P"]
                    + parameters["A_P"] * (time - parameters["tf_P"])
                )
            else:
                value = 1e-4  # for non-active elements

        return value

    @staticmethod
    def von_mises_from_mandel(mandel_stress: np.ndarray) -> np.ndarray:
        """
        Compute von Mises stress from stress tensor in Mandel notation.

        Args:
            mandel_stress: ndarray of shape (n_points, 6), in Mandel notation

        Returns:
            von_mises: ndarray of shape (n_points,)
        """

        mandel_stress = mandel_stress.reshape(-1, 6)

        s = mandel_stress.copy()

        # Compute trace of stress tensor
        trace = s[:, 0] + s[:, 1] + s[:, 2]

        # Deviatoric part: s_ij = sigma_ij - (1/3) * trace * delta_ij
        s[:, 0] -= trace / 3
        s[:, 1] -= trace / 3
        s[:, 2] -= trace / 3

        # Compute von Mises stress
        vm_squared = (
                s[:, 0] ** 2
                + s[:, 1] ** 2
                + s[:, 2] ** 2
                + s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2
        )

        return np.sqrt(1.5 * vm_squared)
