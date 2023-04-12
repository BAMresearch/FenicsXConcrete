import dolfinx as df
import ufl

from fenicsxconcrete.finite_element_problem.base_material import MaterialProblem
from fenicsxconcrete.helper import project
from fenicsxconcrete.sensor_definition.base_sensor import PointSensor
from fenicsxconcrete.unit_registry import ureg


class StrainSensor(PointSensor):
    """A sensor that measures strain at a specific point

    Attributes:
        data: list of measured values
        time: list of time stamps
        units : pint definition of the base unit a sensor returns
        name : name of the sensor, default is class name, but can be changed
        where: location where the value is measured
    """

    def measure(self, problem: MaterialProblem, t: float = 1.0) -> None:
        """
        The strain value at the defined point is added to the data list,
        as well as the time t to the time list

        Arguments:
            problem : FEM problem object
            t : time of measurement for time dependent problems, default is 1
        """

        # project stress onto visualization space
        if not hasattr(problem, "strain"):
            self.logger.debug("strain not defined in problem - needs to compute stress first")
            strain = project(
                problem.epsilon(problem.displacement),  # stress fct from problem
                df.fem.TensorFunctionSpace(problem.experiment.mesh, ("Lagrange", 1)),  # tensor space
                ufl.dx,
            )
        else:
            # TODO: I cannot test this lines, yet (comment: Annika)
            #       why is it implemented??? how do you know it works? (comment: Erik)
            strain = project(problem.strain, problem.visu_space_T, problem.rule.dx)

        # finding the cells corresponding to the point
        bb_tree = df.geometry.BoundingBoxTree(problem.experiment.mesh, problem.experiment.mesh.topology.dim)
        cells = []

        # Find cells whose bounding-box collide with the points
        cell_candidates = df.geometry.compute_collisions(bb_tree, [self.where])

        # Choose one of the cells that contains the point
        colliding_cells = df.geometry.compute_colliding_cells(problem.experiment.mesh, cell_candidates, [self.where])
        if len(colliding_cells.links(0)) > 0:
            cells.append(colliding_cells.links(0)[0])

        # adding correct units to stress
        strain_data = strain.eval([self.where], cells)

        self.data.append(strain_data)
        self.time.append(t)

    @staticmethod
    def base_unit() -> ureg:
        """Defines the base unit of this sensor

        Returns:
            the base unit as pint unit object
        """
        return ureg("")
