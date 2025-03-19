from collections.abc import Callable
from pathlib import Path

import basix
import dolfinx as df
import numpy as np
import pint
import ufl
from fenics_constitutive import Constraint, IncrSmallStrainModel, build_history, ufl_mandel_strain
from mpi4py import MPI
from petsc4py import PETSc

from fenicsxconcrete.experimental_setup import AmMultipleLayers, Experiment
from fenicsxconcrete.finite_element_problem.base_material import MaterialProblem, QuadratureFields, SolutionFields
from fenicsxconcrete.util import QuadratureEvaluator, QuadratureRule, project, ureg


class ConcreteAMFC(MaterialProblem):
    """A class for additive manufacturing models

    - including pseudo density approach for element activation -> set_initial_path == negative time when element will be activated
    - time incremental weak form (in case of density load increments are computed automatic, otherwise user controlled)
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
        if not material or material == "LinearElasticityModel":
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
        else:
            raise ValueError("material law not known")

        return experiment, {**parameters, **model_parameters}

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
        # body_force_fct = self.experiment.create_body_force # TODO temp delete
        body_force_fct = self.experiment.create_body_force_am  # with element activation

        # define problem:
        self.mechanics_problem = ProblemAM(
            law, self.fields.displacement, bcs, body_force_fct, q_degree=self.p["q_degree"]
        )
        self.mechanics_problem._time = self.p["dt"]

        # additional output fields
        self.rule = QuadratureRule(cell_type=self.mesh.ufl_cell(), degree=self.p["q_degree"])

        # self.q_fields = QuadratureFields(
        #     measure=self.rule.dx,
        #     plot_space_type=("CG", self.p["degree"] - 1),
        #     mandel_stress=self.mechanics_problem.stress_1,
        # )
        try:
            self.q_fields = QuadratureFields(
                measure=self.rule.dx,
                plot_space_type=("CG", 1),
                mandel_stress=self.mechanics_problem.stress_1,
                history_scalar=self.mechanics_problem._history_1[-1]["alpha"],
            )
            self.a_plot = True
        except:
            self.q_fields = QuadratureFields(
                measure=self.rule.dx,
                plot_space_type=("CG", 1),
                mandel_stress=self.mechanics_problem.stress_1,
            )
            self.a_plot = False

        self.mandel_stress_dim = law.stress_strain_dim  # for sensor
        self.hist_a = 1

        # additional stuff/output field for activation or specific output
        self.modulus = self.mechanics_problem.modulus
        self.density_time = self.mechanics_problem.density_time
        # array describing path time per quadrature point
        self.q_array_path_time = np.zeros_like(self.density_time.x.array[:])  # zero as default

        # setting up the solver
        self.mechanics_solver = df.nls.petsc.NewtonSolver(MPI.COMM_WORLD, self.mechanics_problem)
        self.mechanics_solver.atol = 1e-9
        self.mechanics_solver.rtol = 1e-8
        self.mechanics_solver.report = True

        # for paraview stress output
        # vector space
        self.plot_space_stress = df.fem.VectorFunctionSpace(
            self.experiment.mesh, self.q_fields.plot_space_type, dim=self.mandel_stress_dim
        )
        self.plot_space_alpha = df.fem.VectorFunctionSpace(
            self.experiment.mesh, self.q_fields.plot_space_type, dim=self.hist_a
        )

    def solve(self) -> None:
        """time incremental solving !"""

        self.update_time()  # set t+dt

        self.logger.info(f"solve for t: {self.time}")
        self.logger.info(f"CHECK if external loads are applied as incremental loads e.g. delta_u(t)!!!")

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

        self.mechanics_problem.update()

        # get sensor data
        self.compute_residuals()  # for residual sensor
        for sensor_name in self.sensors:
            # go through all sensors and measure
            self.sensors[sensor_name].measure(self)

    def compute_residuals(self) -> None:
        """defines what to do, to compute the residuals. Called in solve for sensors"""

        self.residual = self.mechanics_problem.R_form

    def update_material_parameters(self) -> None:
        """update material parameters at each quadrature point according time based on path_time"""

        # print(self.material_law.__name__)

        # print(self.density_time.x.array[:])
        # print('pd min max', self.density_time.x.array[:].min(), self.density_time.x.array[:].max())
        # print('num active elements', len(np.where(self.density_time.x.array[:] > 0)[0]))
        # print('num active elements 1', len(np.where(self.density_time.x.array[:] == 1)[0]))
        # print('num active elements 0.5', len(np.where(self.density_time.x.array[:] == 0.5)[0]))
        # # input()

        # compute material parameters for time t
        if self.material_law.__name__ == "LinearElasticityModel":
            time_params = ["E"]
            p_values = self.get_params_gp(time_params)

            # in the linear model we adapt the factor of the youngs modulus
            self.mechanics_problem.laws[0][0].factor = p_values["E"] / self.p["E"]

            # # store E just for access since material law dependent do it here and not in ProblemAM
            self.mechanics_problem.modulus.x.array[:] = self.mechanics_problem.laws[0][0].factor
            self.mechanics_problem.modulus.x.scatter_forward()

        elif self.material_law.__name__ == "VonMises3D":
            # no changing in the moment
            # time_params = ['p_ka', 'p_mu', 'p_y0']
            # p_values = self.get_params_gp(time_params)
            #
            # self.mechanics_problem.laws[0][0].p_ka = p_values['E0']
            # self.mechanics_problem.laws[0][0].p_mu = p_values['E1']
            # self.mechanics_problem.laws[0][0].p_y0 = p_values['tau']

            # # store bulk modulus just for access since material law dependent do it here and not in ProblemAM
            self.mechanics_problem.modulus.x.array[:] = self.mechanics_problem.laws[0][0].p_ka
            self.mechanics_problem.modulus.x.scatter_forward()
        #
        elif self.material_law.__name__ == "SpringKelvinModel" or self.material_law.__name__ == "SpringMaxwellModel":
            # parameters which can vary over time [E0,E1,tau]
            time_params = ["E0", "E1", "tau"]
            p_values = self.get_params_gp(time_params)

            self.mechanics_problem.laws[0][0].E0 = p_values["E0"]
            self.mechanics_problem.laws[0][0].E1 = p_values["E1"]
            self.mechanics_problem.laws[0][0].tau = p_values["tau"]
            self.mechanics_problem.laws[0][0].factor_E0 = p_values["E0"] / self.p["E0"]
            self.mechanics_problem.laws[0][0].factor_E1 = p_values["E1"] / self.p["E1"]

            # # store E0 just for access since material law dependent do it here and not in ProblemAM
            self.mechanics_problem.modulus.x.array[:] = self.mechanics_problem.laws[0][0].E0
            self.mechanics_problem.modulus.x.scatter_forward()

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
            )  # linear ramp #TODO check

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

        # write further fields
        sigma_plot = project(self.q_fields.mandel_stress, self.plot_space_stress, self.rule.dx)
        sigma_plot.name = "Stress"

        D_plot = project(
            self.density_time, df.fem.FunctionSpace(self.mesh, self.q_fields.plot_space_type), self.rule.dx
        )
        D_plot.name = "Density"
        self.density_plot = D_plot

        if self.a_plot:
            A_plot = project(self.q_fields.history_scalar, self.plot_space_alpha, self.rule.dx)
            A_plot.name = "Alpha"

        # xdmf
        with df.io.XDMFFile(self.mesh.comm, self.pv_output_file, "a") as f:
            f.write_function(self.fields.displacement, self.time)
            f.write_function(sigma_plot, self.time)
            f.write_function(D_plot, self.time)
            if self.a_plot:
                f.write_function(A_plot, self.time)

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


class ProblemAM(df.fem.petsc.NonlinearProblem):
    """general small strain incremental problem for additive manufacturing
        similar to the IncrSmallStrainProblem in fenics-constitutive

        material law as given
        standard weak form with body force and given bcs
        time/temperature dependent material parameters

    Args:
        mesh : The mesh.
        parameters : Dictionary of material parameters.
        rule: The quadrature rule.
        u: displacement fct
        bc: array of Dirichlet boundaries
        body_force: function of creating body force
         q_degree: The quadrature degree (Polynomial degree which the quadrature rule needs to integrate exactly).
        form_compiler_options: The options for the form compiler.
        jit_options: The options for the JIT compiler.
    """

    def __init__(
        self,
        laws: IncrSmallStrainModel,
        u: df.fem.Function,
        bcs: list[df.fem.DirichletBCMetaClass],
        body_force_fct: Callable,
        q_degree: int = 1,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
    ):
        mesh = u.function_space.mesh
        map_c = mesh.topology.index_map(mesh.topology.dim)
        num_cells = map_c.size_local + map_c.num_ghosts
        cells = np.arange(0, num_cells, dtype=np.int32)
        assert isinstance(laws, IncrSmallStrainModel)

        self.laws = [(laws, cells)]
        constraint = self.laws[0][0].constraint

        gdim = mesh.ufl_cell().geometric_dimension()
        assert constraint.geometric_dim() == gdim, "Geometric dimension mismatch between mesh and laws"

        QVe = ufl.VectorElement(
            "Quadrature",
            mesh.ufl_cell(),
            q_degree,
            quad_scheme="default",
            dim=constraint.stress_strain_dim(),
        )
        QTe = ufl.TensorElement(
            "Quadrature",
            mesh.ufl_cell(),
            q_degree,
            quad_scheme="default",
            shape=(constraint.stress_strain_dim(), constraint.stress_strain_dim()),
        )
        Q_grad_u_e = ufl.TensorElement(
            "Quadrature",
            mesh.ufl_cell(),
            q_degree,
            quad_scheme="default",
            shape=(gdim, gdim),
        )
        QV = df.fem.FunctionSpace(mesh, QVe)
        QT = df.fem.FunctionSpace(mesh, QTe)

        self.mesh_update = True
        self.co_rotation = True
        self._del_grad_u = []
        self._stress = []
        self._history_0 = []
        self._history_1 = []
        self._tangent = []

        self._time = 0.0  # time at the end of the increment

        with df.common.Timer("submeshes-and-data-structures"):
            law, cells = self.laws[0]

            # subspace for grad u
            Q_grad_u_space = df.fem.FunctionSpace(mesh, Q_grad_u_e)
            self._del_grad_u.append(df.fem.Function(Q_grad_u_space))

            # Spaces for history
            history_0 = build_history(law, mesh, q_degree)
            history_1 = {key: fn.copy() for key, fn in history_0.items()} if isinstance(history_0, dict) else history_0
            self._history_0.append(history_0)
            self._history_1.append(history_1)

        self.stress_0 = df.fem.Function(QV)
        self.stress_1 = df.fem.Function(QV)
        self.tangent = df.fem.Function(QT)

        # additional field for modulus changing over space and time
        Qs = ufl.FiniteElement(
            "Quadrature",
            mesh.ufl_cell(),
            q_degree,
            quad_scheme="default",
        )
        s_space = df.fem.FunctionSpace(mesh, Qs)
        self.modulus = df.fem.Function(s_space, name="modulus")  # one material parameter
        self.density_time = df.fem.Function(s_space, name="density")

        # define forms
        u_, du = ufl.TestFunction(u.function_space), ufl.TrialFunction(u.function_space)

        self.metadata = {"quadrature_degree": q_degree, "quadrature_scheme": "default"}
        self.dxm = ufl.dx(metadata=self.metadata)

        self.R_form = ufl.inner(ufl_mandel_strain(u_, constraint), self.stress_1) * self.dxm

        # apply body force
        # body_force_form = body_force_fct(u_) # ohne activierung
        rule = QuadratureRule(cell_type=mesh.ufl_cell(), degree=q_degree)
        body_force_form = body_force_fct(u_, self.density_time, rule)

        if body_force_form:
            self.R_form -= body_force_form

        # maybe also external forces?

        self.dR_form = (
            ufl.inner(
                ufl_mandel_strain(du, constraint),
                ufl.dot(self.tangent, ufl_mandel_strain(u_, constraint)),
            )
            * self.dxm
        )

        self._u = u
        self._u0 = u.copy()
        self._bcs = bcs
        self._form_compiler_options = form_compiler_options
        self._jit_options = jit_options

        basix_celltype = getattr(basix.CellType, mesh.topology.cell_type.name)
        self.q_points, _ = basix.make_quadrature(basix_celltype, q_degree)

        self.del_grad_u_expr = df.fem.Expression(ufl.nabla_grad(self._u - self._u0), self.q_points)

    @property
    def a(self) -> df.fem.FormMetaClass:
        """Compiled bilinear form (the Jacobian form)"""

        if not hasattr(self, "_a"):
            # ensure compilation of UFL forms
            super().__init__(
                self.R_form,
                self._u,
                self._bcs,
                self.dR_form,
                form_compiler_options=self._form_compiler_options if self._form_compiler_options is not None else {},
                jit_options=self._jit_options if self._jit_options is not None else {},
            )

        return self._a

    def form(self, x: PETSc.Vec) -> None:
        """This function is called before the residual or Jacobian is
        computed. This is usually used to update ghost values, but here
        we use it to update the stress, tangent and history.

        Args:
            x: The vector containing the latest solution

        """
        super().form(x)

        assert (
            x.array.data == self._u.vector.array.data
        ), "The solution vector must be the same as the one passed to the MechanicsProblem"

        if self.mesh_update:
            # print('mesh update is on')
            dim = self._u.function_space.mesh.topology.dim
            # print('check', len(self._u.function_space.mesh.geometry.x[:]), len(self._u.x.array[:]))
            if len(self._u.function_space.mesh.geometry.x[:]) * dim != len(self._u.x.array[:]):

                V_CG = df.fem.VectorFunctionSpace(self._u.function_space.mesh, ("CG", 1))
                u_CG0 = df.fem.Function(V_CG)
                u_CG = df.fem.Function(V_CG)

                u_CG0.interpolate(self._u0)
                u_CG.interpolate(self._u)
                midpoint_displacement = 0.5 * (u_CG.x.array - u_CG0.x.array)

            else:
                midpoint_displacement = 0.5 * (self._u.x.array - self._u0.x.array)

            self._u.function_space.mesh.geometry.x[:] += midpoint_displacement.reshape(-1, 3)

        law, cells = self.laws[0]
        with df.common.Timer("strain_evaluation"):
            self.del_grad_u_expr.eval(cells, self._del_grad_u[0].x.array.reshape(cells.size, -1))

        with df.common.Timer("stress_evaluation"):
            self.stress_1.x.array[:] = self.stress_0.x.array
            stress_input = self.stress_1.x.array
            if self.co_rotation:
                self.stress_rotate(del_grad_u=self._del_grad_u[0].x.array, mandel_stress=stress_input)
            history_input = None
            if isinstance(law.history_dim, int):
                self._history_1[0].x.array[:] = self._history_0[0].x.array
                history_input = self._history_1[0].x.array
            elif isinstance(law.history_dim, dict):
                history_input = {}
                for key in law.history_dim:
                    self._history_1[0][key].x.array[:] = self._history_0[0][key].x.array
                    history_input[key] = self._history_1[0][key].x.array
            law.evaluate(
                self._time,
                self._del_grad_u[0].x.array,
                stress_input,
                self.tangent.x.array,
                history_input,
            )

        if self.mesh_update:
            self._u.function_space.mesh.geometry.x[:] -= midpoint_displacement.reshape(-1, 3)

        self.stress_1.x.scatter_forward()
        self.tangent.x.scatter_forward()

    def update(self) -> None:
        """
        Update the current displacement, stress and history.
        """

        if self.mesh_update:

            # Update to current configuration
            dim = self._u.function_space.mesh.topology.dim
            if len(self._u.function_space.mesh.geometry.x[:]) * dim != len(self._u.x.array[:]):

                V_CG = df.fem.VectorFunctionSpace(self._u.function_space.mesh, ("CG", 1))
                u_CG0 = df.fem.Function(V_CG)
                u_CG = df.fem.Function(V_CG)

                u_CG0.interpolate(self._u0)
                u_CG.interpolate(self._u)

                current_displacement = u_CG.x.array - u_CG0.x.array
            else:
                current_displacement = self._u.x.array - self._u0.x.array

            self._u.function_space.mesh.geometry.x[:] += current_displacement.reshape(-1, 3)

        self._u0.x.array[:] = self._u.x.array
        self._u0.x.scatter_forward()

        self.stress_0.x.array[:] = self.stress_1.x.array
        self.stress_0.x.scatter_forward()

        law, _ = self.laws[0]
        match law.history_dim:
            case int():
                self._history_0[0].x.array[:] = self._history_1[0].x.array
                self._history_0[0].x.scatter_forward()
            case None:
                pass
            case dict():
                for key in law.history_dim:
                    self._history_0[0][key].x.array[:] = self._history_1[0][key].x.array
                    self._history_0[0][key].x.scatter_forward()

    def stress_rotate(self, del_grad_u, mandel_stress):
        # TODO the stress that we get here is mandel stress already. convert it into 3x3 form using appropriate expressions
        # I2 = np.zeros((3,3), dtype=np.float64)  # Identity of rank 2 tensor
        # I2[0, 0] = 1.0
        # I2[1, 1] = 1.0
        # I2[2, 2] = 1.0
        I2 = np.eye(3, 3)
        shape = int(np.shape(del_grad_u)[0] / 9)

        mandel_stress = mandel_stress.reshape(-1, 6)

        # print(np.shape(mandel_stress))

        stress = np.zeros((shape, 3, 3), dtype=np.float64)

        stress[:, 0, 0] = mandel_stress[:, 0]
        stress[:, 1, 1] = mandel_stress[:, 1]
        stress[:, 2, 2] = mandel_stress[:, 2]
        stress[:, 0, 1] = 1 / 2**0.5 * (mandel_stress[:, 3])
        stress[:, 1, 2] = 1 / 2**0.5 * (mandel_stress[:, 4])
        stress[:, 0, 2] = 1 / 2**0.5 * (mandel_stress[:, 5])
        stress[:, 1, 0] = stress[:, 0, 1]
        stress[:, 2, 1] = stress[:, 1, 2]
        stress[:, 2, 0] = stress[:, 0, 2]

        # print(del_grad_u)
        # g = del_grad_u.reshape(-1, 9)
        g = del_grad_u.reshape(shape, 3, 3)
        # print(g)
        # rotated_stress_matrix = []

        for n, eps in enumerate(g):
            # strain_increment = (eps + np.transpose(eps))/2
            rotation_increment = (eps - np.transpose(eps)) / 2
            # print(rotation_increment)
            # print('rotation increment', rotation_increment)
            Q_matrix = I2 + (np.linalg.inv(I2 - 0.5 * rotation_increment)) @ rotation_increment
            rot_stress = Q_matrix.T @ stress[n, :, :] @ Q_matrix
            # print(Q_matrix)
            stress[n, :, :] = rot_stress
            # rotated_stress_matrix.append(rot_stress)

        # rotated_stress_matrix = np.array(rotated_stress_matrix)
        # print(np.shape(rotated_stress_matrix))
        rotated_stress_mandel = np.zeros((shape, 6), dtype=np.float64)

        rotated_stress_mandel[:, 0] = stress[:, 0, 0]
        rotated_stress_mandel[:, 1] = stress[:, 1, 1]
        rotated_stress_mandel[:, 2] = stress[:, 2, 2]
        rotated_stress_mandel[:, 3] = 2**0.5 * stress[:, 0, 1]
        rotated_stress_mandel[:, 4] = 2**0.5 * stress[:, 1, 2]
        rotated_stress_mandel[:, 5] = 2**0.5 * stress[:, 0, 2]

        # print('mandel stress rotated ################',rotated_stress_mandel)
        # mandel_stress = mandel_stress.flatten()
        mandel_stress[:, :] = rotated_stress_mandel
