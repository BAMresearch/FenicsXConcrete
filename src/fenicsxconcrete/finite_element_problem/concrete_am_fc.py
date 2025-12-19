from collections.abc import Callable

import basix
import dolfinx as df
import numpy as np
import pint
import ufl
from dolfinx.nls.petsc import NewtonSolver
from fenics_constitutive import IncrSmallStrainModel, StressStrainConstraint, build_history, ufl_mandel_strain
from mpi4py import MPI
from petsc4py import PETSc

from fenicsxconcrete.experimental_setup import AmMultipleLayers, Experiment
from fenicsxconcrete.finite_element_problem.base_material import MaterialProblem, QuadratureFields, SolutionFields
from fenicsxconcrete.util import QuadratureRule, project, ureg

from fenicsxconcrete.finite_element_problem.material_for_am_fc import LinearElasticityModel


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

        # define problem:

        # material law based on fenics constitutive interface
        try:
            law = self.material_law(self.p, constraint=StressStrainConstraint.FULL)
        except:
            law = self.material_law(self.p)
            assert law.constraint == StressStrainConstraint.FULL

        # boundaries
        bcs = self.experiment.create_displacement_boundary(self.V)
        body_force_fct = self.experiment.create_body_force_am  # with element activation

        # define problem:
        self.mechanics_problem = ProblemAM(
            law, self.fields.displacement, bcs, body_force_fct, q_degree=self.p["q_degree"], del_t=self.p["dt"])
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
        self.von_mises = self.mechanics_problem.von_mises
        # array describing path time per quadrature point
        self.q_array_path_time = np.zeros_like(self.density_time.x.array[:])  # zero as default

        # setting up the solver
        self.mechanics_solver = NewtonSolver(MPI.COMM_WORLD, self.mechanics_problem)
        self.mechanics_solver.atol = 1e-9
        self.mechanics_solver.rtol = 1e-8
        self.mechanics_solver.report = True


        # for paraview stress output
        # vector space
        self.plot_space_stress = df.fem.functionspace(
           self.experiment.mesh, (self.q_fields.plot_space_type[0], self.q_fields.plot_space_type[1], (self.mandel_stress_dim,))
        )
        # history alpha space
        self.plot_space_alpha = df.fem.functionspace(
            self.experiment.mesh, self.q_fields.plot_space_type)
        # # TODO check which one scalar or vector
        # self.plot_space_alpha = df.fem.functionspace(
        #     self.experiment.mesh, (self.q_fields.plot_space_type[0], self.q_fields.plot_space_type[1], (self.hist_a,))
        # )

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
            # changing parameters
            time_params = ["p_ka", "p_mu", "p_y0", "p_y00", "p_w"]
            p_values = self.get_params_gp(time_params)
            #
            self.mechanics_problem.laws[0][0].p_ka = p_values["p_ka"]
            self.mechanics_problem.laws[0][0].p_mu = p_values["p_mu"]
            self.mechanics_problem.laws[0][0].p_y0 = p_values["p_y0"]
            self.mechanics_problem.laws[0][0].p_y00 = p_values["p_y00"]
            self.mechanics_problem.laws[0][0].p_w = p_values["p_w"]

            # # store bulk modulus just for access since material law dependent do it here and not in ProblemAM
            self.mechanics_problem.modulus.x.array[:] = self.mechanics_problem.laws[0][0].p_ka
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
        # density_plot = df.fem.Function(self.plot_space_alpha, name="Density")
        # project(self.density_time, self.plot_space_alpha, ufl.dx, density_plot)
        # density_plot.x.scatter_forward()

        Q0 = df.fem.functionspace(self.mesh, ("DG", 0))
        VM_plot = project(self.von_mises, Q0, self.rule.dx)
        VM_plot.name = "Von-Mises_DG0"

        if self.a_plot:
            alpha_plot = project(self.q_fields.history_scalar, self.plot_space_alpha, self.rule.dx)
            alpha_plot.name = "Alpha"
            # alpha_plot = df.fem.Function(self.plot_space_alpha, name="Alpha")   
            # project(self.q_fields.history_scalar, self.plot_space_alpha, ufl.dx, alpha_plot)
            # alpha_plot.x.scatter_forward()
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
        bcs: list[df.fem.DirichletBC],
        body_force_fct: Callable,
        q_degree: int = 1,
        del_t: float=1.0,
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

        gdim = mesh.geometry.dim
        assert constraint.geometric_dim == gdim, "Geometric dimension mismatch between mesh and laws"

        QVe = basix.ufl.quadrature_element(
            mesh.topology.cell_name(),
            value_shape=(constraint.stress_strain_dim,),
            degree=q_degree,
        )
        QTe = basix.ufl.quadrature_element(
            mesh.topology.cell_name(),
            value_shape=(
                constraint.stress_strain_dim,
                constraint.stress_strain_dim,
            ),
            degree=q_degree,
        )
        Q_grad_u_e = basix.ufl.quadrature_element(mesh.topology.cell_name(), value_shape=(gdim, gdim), degree=q_degree)
        QV = df.fem.functionspace(mesh, QVe)
        QT = df.fem.functionspace(mesh, QTe)

        self.mesh_update = True  # DIFF to FC solver
        self.co_rotation = True  # DIFF to FC solver
        self._del_grad_u = []
        self._stress = []
        self._history_0 = []
        self._history_1 = []
        self._tangent = []

        self._del_t = del_t  # time increment
        self._time = 0  # global time will be updated in the update method

        with df.common.Timer("data-structures"):
            law, cells = self.laws[0]

            # space for grad u
            Q_grad_u_space = df.fem.functionspace(mesh, Q_grad_u_e)
            self._del_grad_u = df.fem.Function(Q_grad_u_space)

            # space for tangent
            QT_space = df.fem.functionspace(mesh, QTe)
            self._tangent = df.fem.Function(QT_space)

            # Spaces for history
            history_0 = build_history(law, mesh, q_degree)
            history_1 = {key: fn.copy() for key, fn in history_0.items()} if isinstance(history_0, dict) else history_0
            self._history_0 = history_0
            self._history_1 = history_1

        self.stress_0 = df.fem.Function(QV)
        self.stress_1 = df.fem.Function(QV)
        self.tangent = df.fem.Function(QT)

        ### DIFF to FC solver
        # additional field for modulus changing over space and time
        QSe = basix.ufl.quadrature_element(
            mesh.topology.cell_name(),
            value_shape=(),  # scalar
            degree=q_degree,
        )
        s_space = df.fem.functionspace(mesh, QSe)
        self.modulus = df.fem.Function(s_space, name="modulus")  # one material parameter
        self.density_time = df.fem.Function(s_space, name="density")
        self.von_mises = df.fem.Function(s_space, name="von_mises")
        ###

        # define forms
        u_, du = ufl.TestFunction(u.function_space), ufl.TrialFunction(u.function_space)

        self.metadata = {"quadrature_degree": q_degree, "quadrature_scheme": "default"}
        self.dxm = ufl.dx(metadata=self.metadata)

        self.R_form = ufl.inner(ufl_mandel_strain(u_, constraint), self.stress_1) * self.dxm

        ### DIFF to FC solver
        # apply body force
        # body_force_form = body_force_fct(u_) # ohne activierung
        rule = QuadratureRule(cell_type=mesh.ufl_cell(), degree=q_degree)
        body_force_form = body_force_fct(u_, self.density_time, rule)

        if body_force_form:
            self.R_form -= body_force_form
        ###

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
    def a(self) -> df.fem.Form:
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

    @df.common.timed("constitutive-form-evaluation")
    def form(self, x: PETSc.Vec) -> None:
        """This function is called before the residual or Jacobian is
        computed. This is usually used to update ghost values, but here
        we use it to update the stress, tangent and history.

        Args:
            x: The vector containing the latest solution

        """
        super().form(x)
        # this copies the data from the vector x to the function _u
        x.copy(self._u.x.petsc_vec)
        self._u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        ### DIFF to FC solver
        if self.mesh_update:
            # print('mesh update is on')
            dim = self._u.function_space.mesh.topology.dim
            # print('check', len(self._u.function_space.mesh.geometry.x[:]), len(self._u.x.array[:]))
            if len(self._u.function_space.mesh.geometry.x[:]) * dim != len(self._u.x.array[:]):
                V_CG = df.fem.functionspace(self._u.function_space.mesh, ("CG", 1, (dim,)))
                u_CG0 = df.fem.Function(V_CG)
                u_CG = df.fem.Function(V_CG)

                u_CG0.interpolate(self._u0)
                u_CG.interpolate(self._u)
                midpoint_displacement = 0.5 * (u_CG.x.array - u_CG0.x.array)

            else:
                midpoint_displacement = 0.5 * (self._u.x.array - self._u0.x.array)

            self._u.function_space.mesh.geometry.x[:] += midpoint_displacement.reshape(-1, 3)
        ###

        law, cells = self.laws[0]
        with df.common.Timer("strain_evaluation"):
            self._del_grad_u.interpolate(
                self.del_grad_u_expr,
                cells0=cells,
                cells1=np.arange(cells.size, dtype=np.int32),
            )
            self._del_grad_u.x.scatter_forward()

        with df.common.Timer("stress_evaluation"):
            self.stress_1.x.array[:] = self.stress_0.x.array
            self.stress_1.x.scatter_forward()
            stress_input = self.stress_1.x.array
            tangent_input = self.tangent.x.array

            ### DIFF to FC solver
            if self.co_rotation:
                self.stress_rotate(del_grad_u=self._del_grad_u.x.array, mandel_stress=stress_input)
            ###

            history_input = None
            if law.history_dim is not None:
                history_input = {}
                for key in law.history_dim:
                    self._history_1[key].x.array[:] = self._history_0[key].x.array
                    history_input[key] = self._history_1[key].x.array
            with df.common.Timer("constitutive-law-evaluation"):
                law.evaluate(
                    self._time,
                    self._del_t,
                    self._del_grad_u.x.array,
                    stress_input,
                    tangent_input,
                    history_input,
                )

        ### DIFF to FC solver
        if self.mesh_update:
            self._u.function_space.mesh.geometry.x[:] -= midpoint_displacement.reshape(-1, 3)
        ###

        self.stress_1.x.scatter_forward()
        self.tangent.x.scatter_forward()
        self.von_mises.x.array[:] = self.von_mises_from_mandel(self.stress_1.x.array)
        self.von_mises.x.scatter_forward()

    def update(self) -> None:
        """
        Update the current displacement, stress and history.
        """

        ### DIFF to FC solver
        if self.mesh_update:
            # Update to current configuration
            dim = self._u.function_space.mesh.topology.dim
            if len(self._u.function_space.mesh.geometry.x[:]) * dim != len(self._u.x.array[:]):
                V_CG = df.fem.functionspace(self._u.function_space.mesh, ("CG", 1, (dim,)))
                u_CG0 = df.fem.Function(V_CG)
                u_CG = df.fem.Function(V_CG)

                u_CG0.interpolate(self._u0)
                u_CG.interpolate(self._u)

                current_displacement = u_CG.x.array - u_CG0.x.array
            else:
                current_displacement = self._u.x.array - self._u0.x.array

            self._u.function_space.mesh.geometry.x[:] += current_displacement.reshape(-1, 3)
        ###

        self._u0.x.array[:] = self._u.x.array
        self._u0.x.scatter_forward()

        self.stress_0.x.array[:] = self.stress_1.x.array
        self.stress_0.x.scatter_forward()

        law, _ = self.laws[0]
        if law.history_dim is not None:
            for key in law.history_dim:
                self._history_0[key].x.array[:] = self._history_1[key].x.array
                self._history_0[key].x.scatter_forward()

        # time update
        self._time += self._del_t

    ###DIFF to FC solver
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

    def von_mises_from_mandel(self, mandel_stress):
        """
        Compute von Mises stress from stress tensor in Mandel notation.

        Args:
            mandel_stress: ndarray of shape (n_points, 6), in Mandel notation

        Returns:
            von_mises: ndarray of shape (n_points,)
        """

        I2 = np.eye(3, 3)
        # shape = int(np.shape(self.mechanics_problem._del_grad_u)[0] / 9)

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

        ans = np.sqrt(1.5 * vm_squared)

        return ans
