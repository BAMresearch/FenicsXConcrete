import dolfinx as df
import pint
from fenics_constitutive import Constraint, IncrSmallStrainModel, IncrSmallStrainProblem
from mpi4py import MPI

from fenicsxconcrete.experimental_setup import Experiment, SimpleCube
from fenicsxconcrete.finite_element_problem.base_material import MaterialProblem, QuadratureFields, SolutionFields
from fenicsxconcrete.util import QuadratureRule, project, ureg


class FenicsConstitutive(MaterialProblem):
    """A class for material laws defined in the fenics-constitutive interface

    - connects material problem class with incremental small strain problem from fenics-constitutive

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
            material: material law as IncSmallStrainModel from fenics-constitutive
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
    def default_parameters() -> tuple[Experiment, dict[str, pint.Quantity]]:
        """Static method that returns a set of default parameters for the selected nonlinear problem.

        Returns:
            The default experiment instance and the default parameters as a dictionary.

        """

        # default experiment
        experiment = SimpleCube(SimpleCube.default_parameters())

        # default parameters according given nonlinear problem
        parameters = {
            # general parameters
            "rho": 2070 * ureg("kg/m^3"),  # density
            "g": 9.81 * ureg("m/s^2"),  # gravity
            # general model parameters
            "degree": 2 * ureg(""),  # polynomial degree
            "q_degree": 2 * ureg(""),  # quadrature rule
            "dt": 1.0 * ureg("s"),  # time step
            # material parameters
            # ... - according to chosen material law!
        }

        return experiment, parameters

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

        # problem
        self.mechanics_problem = IncrSmallStrainProblem(
            law, self.fields.displacement, bcs, q_degree=self.p["q_degree"]
        )
        # add external force and body force not implemented on IncrSmallStrainProblem
        external_force = self.experiment.create_force_boundary(self.V)
        if external_force:
            self.mechanics_problem.R_form += external_force

        body_force = self.experiment.create_body_force(self.V)
        if body_force:
            self.mechanics_problem.R_form -= body_force  # TODO check sign!!

        # additional output fields
        self.rule = QuadratureRule(cell_type=self.mesh.ufl_cell(), degree=self.p["q_degree"])
        self.q_fields = QuadratureFields(
            measure=self.rule.dx,
            plot_space_type=("CG", self.p["degree"] - 1),
            mandel_stress=self.mechanics_problem.stress_1,  # vector space!! not working with stress_sensor
        )
        self.mandel_stress_dim = law.stress_strain_dim  # for sensor

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

    def solve(self) -> None:
        """time incremental solving !"""

        self.update_time()  # set t+dt

        self.logger.info(f"solve for t: {self.time}")
        self.logger.info(f"CHECK if external loads are applied as incremental loads e.g. delta_u(t)!!!")

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

    def pv_plot(self) -> None:
        """creates paraview output at given time step"""

        self.logger.info(f"create pv plot for t: {self.time}")

        # write further fields
        sigma_plot = project(self.q_fields.mandel_stress, self.plot_space_stress, self.rule.dx)
        sigma_plot.name = "Stress"
        #
        with df.io.XDMFFile(self.mesh.comm, self.pv_output_file, "a") as f:
            f.write_function(self.fields.displacement, self.time)
            f.write_function(sigma_plot, self.time)
