import dolfinx as df
import ufl

from fenicsxconcrete.finite_element_problem.base_material import MaterialProblem
from fenicsxconcrete.helper import project
from fenicsxconcrete.sensor_definition.base_sensor import Sensor
from fenicsxconcrete.unit_registry import ureg


class StressSensor(Sensor):
    """A sensor that measure the stress tensor at a specific point

    Attributes:
        where: location where the value is measured
        date: list of measured values with unit
        time: list of time stamps with unit

    """

    def __init__(self, where: list[list[int | float]]) -> None:
        """initialize object

        Args:
            where: location where the value is measured

        """
        self.where = where
        self.data = []
        self.time = []

    def measure(self, problem: MaterialProblem, t: float = 1.0) -> None:
        """measure stress at given point

        Args:
            problem : FEM problem object
            t : time of measurement for time dependent problems

        """

        # project stress onto visualization space
        if not hasattr(problem, "stress"):
            self.logger.debug("strain not defined in problem - needs to compute stress first")
            stress = project(
                problem.sigma(problem.displacement),  # stress fct from problem
                df.fem.TensorFunctionSpace(problem.experiment.mesh, ("CG", 1)),  # tensor space
                ufl.dx,
            )
        else:
            # TODO: I cannot test this lines, yet
            stress = project(problem.stress, problem.visu_space_T, problem.rule.dx)

        # finding the cells corresponding to the point
        bb_tree = df.geometry.BoundingBoxTree(problem.experiment.mesh, problem.experiment.mesh.topology.dim)
        cells = []

        # Find cells whose bounding-box collide with the points
        cell_candidates = df.geometry.compute_collisions(bb_tree, self.where)

        # Choose one of the cells that contains the point
        colliding_cells = df.geometry.compute_colliding_cells(problem.experiment.mesh, cell_candidates, self.where)
        for i, point in enumerate(self.where):
            if len(colliding_cells.links(i)) > 0:
                cells.append(colliding_cells.links(i)[0])

        # adding correct units to stress
        stress_data = stress.eval(self.where, cells) * ureg("N/m^2")

        self.data.append(stress_data)
        self.time.append(t)
