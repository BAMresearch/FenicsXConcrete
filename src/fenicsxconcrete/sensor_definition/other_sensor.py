import dolfinx as df
import numpy as np
import ufl

from fenicsxconcrete.boundary_conditions.bcs import BoundaryConditions
from fenicsxconcrete.finite_element_problem.base_material import MaterialProblem
from fenicsxconcrete.sensor_definition.base_sensor import Sensor


class TemperatureSensor(Sensor):
    """A sensor that measure temperature at a specific point in celsius"""

    def __init__(self, where: list[list[int | float]]) -> None:
        """
        Arguments:
            where : Point
                location where the value is measured
        """
        self.where = where
        self.data = []
        self.time = []

    def measure(self, problem: MaterialProblem, t: float = 1.0) -> None:
        """
        Arguments:
            problem : FEM problem object
            t : float, optional
                time of measurement for time dependent problems
        """
        T = problem.temperature(self.where) - problem.p.zero_C
        self.data.append(T)
        self.time.append(t)


class MaxTemperatureSensor(Sensor):
    """A sensor that measure the maximum temperature at each timestep"""

    def __init__(self) -> None:
        self.data = [0.0]
        self.time = [0.0]
        self.max = 0.0

    def measure(self, problem: MaterialProblem, t: float = 1.0) -> None:
        """
        Arguments:
            problem : FEM problem object
            t : float, optional
                time of measurement for time dependent problems
        """
        max_T = np.amax(problem.temperature.vector().get_local()) - problem.p.zero_C
        self.data.append(max_T)
        self.data_max(max_T)


class DOHSensor(Sensor):
    """A sensor that measure the degree of hydration at a point"""

    def __init__(self, where: list[list[int | float]]) -> None:
        """
        Arguments:
            where : Point
                location where the value is measured
        """
        self.where = where
        self.data = []
        self.time = []

    def measure(self, problem: MaterialProblem, t: float = 1.0):
        """
        Arguments:
            problem : FEM problem object
            t : float, optional
                time of measurement for time dependent problems
        """
        # get DOH
        # TODO: problem with projected field onto linear mesh!?!
        alpha = problem.degree_of_hydration(self.where)
        self.data.append(alpha)
        self.time.append(t)


class MinDOHSensor(Sensor):
    """A sensor that measure the minimum degree of hydration at each timestep"""

    def __init__(self) -> None:
        self.data = []
        self.time = []

    def measure(self, problem: MaterialProblem, t: float = 1.0) -> None:
        """
        Arguments:
            problem : FEM problem object
            t : float, optional
                time of measurement for time dependent problems
        """
        # get min DOH
        min_DOH = np.amin(problem.q_degree_of_hydration.vector().get_local())
        self.data.append(min_DOH)
        self.time.append(t)


class MaxYieldSensor(Sensor):
    """A sensor that measure the maximum value of the yield function

    A max value > 0 indicates that at some place the stress exceeds the limits"""

    def __init__(self):
        self.data = [0.0]
        self.time = [0.0]
        self.max = 0.0

    def measure(self, problem: MaterialProblem, t: float = 1.0) -> None:
        """
        Arguments:
            problem : FEM problem object
            t : float, optional
                time of measurement for time dependent problems
        """
        max_yield = np.amax(problem.q_yield.vector().get_local())
        self.data.append(max_yield)
        self.time.append(t)
        self.data_max(max_yield)


class ReactionForceSensorBottom(Sensor):
    """A sensor that measure the reaction force at the bottom perpendicular to the surface"""

    def __init__(self) -> None:
        self.data = []
        self.time = []

    def measure(self, problem: MaterialProblem, t: float = 1.0) -> None:
        """
        Arguments:
            problem : FEM problem object
            t : float, optional
                time of measurement for time dependent problems
        """
        # boundary condition
        bottom_surface = problem.experiment.boundary_bottom()

        v_reac = df.fem.Function(problem.V)
        bc_generator = BoundaryConditions(problem.mesh, problem.V)
        if problem.p["dim"] == 2:
            bc_generator.add_dirichlet_bc(
                df.fem.Constant(domain=problem.mesh, c=1.0),
                bottom_surface,
                1,
                "geometrical",
                1,
            )

        elif problem.p["dim"] == 3:
            bc_generator.add_dirichlet_bc(
                df.fem.Constant(domain=problem.mesh, c=1.0),
                bottom_surface,
                2,
                "geometrical",
                2,
            )

        df.fem.set_bc(v_reac.vector, bc_generator.bcs)
        computed_force = -df.fem.assemble_scalar(df.fem.form(ufl.action(problem.residual, v_reac)))

        self.data.append(computed_force)
        self.time.append(t)


class StressSensor(Sensor):
    """A sensor that measure the stress tensor in at a point"""

    def __init__(self, where: list[list[int | float]]) -> None:
        """
        Arguments:
            where : Point
                location where the value is measured
        """
        self.where = where
        self.data = []
        self.time = []

    def measure(self, problem: MaterialProblem, t: float = 1.0) -> None:
        """
        Arguments:
            problem : FEM problem object
            t : float, optional
                time of measurement for time dependent problems
        """
        # get stress
        stress = df.project(
            problem.stress,
            problem.visu_space_T,
            form_compiler_parameters={"quadrature_degree": problem.p.degree},
        )
        self.data.append(stress(self.where))
        self.time.append(t)


class StrainSensor(Sensor):
    """A sensor that measure the strain tensor in at a point"""

    def __init__(self, where: list[list[int | float]]) -> None:
        """
        Arguments:
            where : Point
                location where the value is measured
        """
        self.where = where
        self.data = []
        self.time = []

    def measure(self, problem: MaterialProblem, t: float = 1.0) -> None:
        """
        Arguments:
            problem : FEM problem object
            t : float, optional
                time of measurement for time dependent problems
        """
        # get strain
        strain = df.project(
            problem.strain,
            problem.visu_space_T,
            form_compiler_parameters={"quadrature_degree": problem.p.degree},
        )
        self.data.append(strain(self.where))
        self.time.append(t)
