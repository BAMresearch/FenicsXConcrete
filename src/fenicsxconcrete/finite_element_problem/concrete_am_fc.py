import copy
from collections.abc import Callable
from typing import Type

import basix
import dolfinx as df
import numpy as np
import pint
import ufl
from fenics_constitutive import (
    Constraint,
    IncrSmallStrainModel,
    IncrSmallStrainProblem,
    build_history,
    ufl_mandel_strain,
)
from mpi4py import MPI
from petsc4py import PETSc

from fenicsxconcrete.experimental_setup import AmMultipleLayers, Experiment
from fenicsxconcrete.finite_element_problem.base_material import MaterialProblem, QuadratureFields, SolutionFields
from fenicsxconcrete.util import Parameters, QuadratureEvaluator, QuadratureRule, project, ureg


class ConcreteAMFC(MaterialProblem):
    """A class for additive manufacturing models

    - including pseudo density approach for element activation -> set_initial_path == negative time when element will be activated
    - time incremental weak form (in case of density load increments are computed automatic, otherwise user controlled)
    - material laws from fenics-constitutive (incremental small strain models)

    Attributes:
        nonlinear_problem: the nonlinear problem class of used material law
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
            "parameters": "to be done",
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

        # default parameters according given nonlinear problem #TODO
        parameters = {
            # Material parameter for concrete model
            "rho": 2070 * ureg("kg/m^3"),  # density of fresh concrete
            "g": 9.81 * ureg("m/s^2"),  # gravity
            # other model parameters
            "degree": 2 * ureg(""),  # polynomial degree
            "q_degree": 2 * ureg(""),  # quadrature rule
            "dt": 1.0 * ureg("s"),  # time step
            "load_time": 60 * ureg("s"),  # body force load applied in s
            # plasticity material parameters # TODO: check units
            "p_ka": 175000 * ureg("MPa"),  # bulk modulus
            "p_mu": 80769 * ureg("MPa"), # shear modulus
            "p_y0": 1200 * ureg("MPa"), # initial yield stress
            "p_y00": 2500 * ureg("MPa"), # final yield stress
            "p_w": 200 * ureg(""), # saturation parameter
        }

        return experiment, {**parameters}

    def setup(self) -> None:
        """set up problem"""

        # displacement space (name V required for sensors!)
        self.V = df.fem.VectorFunctionSpace(self.mesh, ("CG", self.p["degree"]))
        self.u = df.fem.Function(self.V)

        # global variables for all AM problems relevant
        self.fields = SolutionFields(displacement=df.fem.Function(self.V, name="displacement"))

        self.rule = QuadratureRule(cell_type=self.mesh.ufl_cell(), degree=self.p["q_degree"])
        self.strain_stress_space = self.rule.create_quadrature_tensor_space(self.mesh, (self.p["dim"], self.p["dim"]))

        self.q_fields = QuadratureFields(
            measure=self.rule.dx,
            plot_space_type=("DG", self.p["degree"] - 1),
            strain=df.fem.Function(self.strain_stress_space, name="strain"),
            stress=df.fem.Function(self.strain_stress_space, name="stress"),
        )

        # material law
        law = self.material_law(self.p, constraint = Constraint.FULL)

        # boundaries
        bcs = self.experiment.create_displacement_boundary(self.V)
        body_force_fct = self.experiment.create_body_force_am

        # problem
        #self.mechanics_problem = Problem_AM(law, self.u, bcs, body_force_fct, q_degree=self.p["q_degree"])
        self.mechanics_problem = IncrSmallStrainProblem(law, self.u, bcs, q_degree=self.p["q_degree"])

        # setting up the solver
        self.mechanics_solver = df.nls.petsc.NewtonSolver(MPI.COMM_WORLD, self.mechanics_problem)
        # self.mechanics_solver.convergence_criterion = "incremental"
        # self.mechanics_solver.atol = 1e-9
        # self.mechanics_solver.rtol = 1e-8
        # self.mechanics_solver.report = True

    def solve(self) -> None:
        """time incremental solving !"""

        self.update_time()  # set t+dt
        # self.update_path()  # set path

        self.logger.info(f"solve for t: {self.time}")
        self.logger.info(f"CHECK if external loads are applied as incremental loads e.g. delta_u(t)!!!")

        # solve problem for current time increment
        self.mechanics_solver.solve(self.u)
        self.mechanics_problem.update() # TODO at which point?

        # update total displacement
        self.fields.displacement.vector.array[:] = self.u.vector.array[:]
        self.fields.displacement.x.scatter_forward()

        # save fields to global problem for sensor output
        # TODO

        # get sensor data
        self.compute_residuals()  # for residual sensor
        for sensor_name in self.sensors:
            # go through all sensors and measure
            self.sensors[sensor_name].measure(self)

    def compute_residuals(self) -> None:
        """defines what to do, to compute the residuals. Called in solve for sensors"""

        self.residual = self.mechanics_problem.R_form

    def pv_plot(self) -> None:
        """creates paraview output at given time step"""

        self.logger.info(f"create pv plot for t: {self.time}")

        # # write further fields
        # sigma_plot = project(
        #     self.mechanics_problem.sigma(self.fields.displacement),
        #     df.fem.TensorFunctionSpace(self.mesh, self.q_fields.plot_space_type),
        #     self.rule.dx,
        # )
        #
        # E_plot = project(
        #     self.mechanics_problem.q_E, df.fem.FunctionSpace(self.mesh, self.q_fields.plot_space_type), self.rule.dx
        # )
        #
        # E_plot.name = "Youngs_Modulus"
        # sigma_plot.name = "Stress"
        #
        with df.io.XDMFFile(self.mesh.comm, self.pv_output_file, "a") as f:
            f.write_function(self.fields.displacement, self.time)
        #     f.write_function(sigma_plot, self.time)
        #     f.write_function(E_plot, self.time)
        #

class Problem_AM(df.fem.petsc.NonlinearProblem):
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
        print("help, i am in init")
        mesh = u.function_space.mesh
        map_c = mesh.topology.index_map(mesh.topology.dim)
        num_cells = map_c.size_local + map_c.num_ghosts
        cells = np.arange(0, num_cells, dtype=np.int32)
        assert isinstance(laws, IncrSmallStrainModel)

        self.laws = [(laws, cells)]
        constraint = self.laws[0][0].constraint

        gdim = mesh.ufl_cell().geometric_dimension()
        assert (
                constraint.geometric_dim() == gdim
        ), "Geometric dimension mismatch between mesh and laws"

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
            history_1 = (
                {key: fn.copy() for key, fn in history_0.items()}
                if isinstance(history_0, dict)
                else history_0
            )
            self._history_0.append(history_0)
            self._history_1.append(history_1)

        self.stress_0 = df.fem.Function(QV)
        self.stress_1 = df.fem.Function(QV)
        self.tangent = df.fem.Function(QT)

        u_, du = ufl.TestFunction(u.function_space), ufl.TrialFunction(u.function_space)

        self.metadata = {"quadrature_degree": q_degree, "quadrature_scheme": "default"}
        self.dxm = ufl.dx(metadata=self.metadata)

        self.R_form = (
                ufl.inner(ufl_mandel_strain(u_, constraint), self.stress_1) * self.dxm
        )
        # # apply body force
        # body_force = body_force_fct(v, self.q_fd, self.rule)
        # if body_force:
        #     self.R_form  -= body_force

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

        self.del_grad_u_expr = df.fem.Expression(
            ufl.nabla_grad(self._u - self._u0), self.q_points
        )

    @property
    def a(self) -> df.fem.FormMetaClass:
        """Compiled bilinear form (the Jacobian form)"""
        print("help, i am in a")
        if not hasattr(self, "_a"):
            # ensure compilation of UFL forms
            super().__init__(
                self.R_form,
                self._u,
                self._bcs,
                self.dR_form,
                form_compiler_options=self._form_compiler_options
                if self._form_compiler_options is not None
                else {},
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
        print("help, i am in form")
        assert (
                x.array.data == self._u.vector.array.data
        ), "The solution vector must be the same as the one passed to the MechanicsProblem"
        law, cells = self.laws[0]
        with df.common.Timer("strain_evaluation"):
            self.del_grad_u_expr.eval(
                cells, self._del_grad_u[0].x.array.reshape(cells.size, -1)
            )

        with df.common.Timer("stress_evaluation"):
            self.stress_1.x.array[:] = self.stress_0.x.array
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
                self.stress_1.x.array,
                self.tangent.x.array,
                history_input,
            )

        self.stress_1.x.scatter_forward()
        self.tangent.x.scatter_forward()

    def update(self) -> None:
        """
        Update the current displacement, stress and history.
        """
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
                    self._history_0[0][key].x.array[:] = self._history_1[0][
                        key
                    ].x.array
                    self._history_0[0][key].x.scatter_forward()