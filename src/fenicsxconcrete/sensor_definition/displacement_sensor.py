import dolfinx as df

from fenicsxconcrete.finite_element_problem.base_material import MaterialProblem
from fenicsxconcrete.sensor_definition.base_sensor import PointSensor
from fenicsxconcrete.unit_registry import ureg


class DisplacementSensor(PointSensor):
    """A sensor that measure displacement at a specific points"""

    def measure(self, problem: MaterialProblem, t: float = 1.0) -> None:
        """
        Arguments:
            problem : FEM problem object
            t : float, optional
                time of measurement for time dependent problems
        """
        # get displacements
        bb_tree = df.geometry.BoundingBoxTree(problem.experiment.mesh, problem.experiment.mesh.topology.dim)
        cells = []

        # Find cells whose bounding-box collide with the points
        cell_candidates = df.geometry.compute_collisions(bb_tree, [self.where])

        # Choose one of the cells that contains the point
        colliding_cells = df.geometry.compute_colliding_cells(problem.experiment.mesh, cell_candidates, [self.where])

        # for i, point in enumerate(self.where):
        if len(colliding_cells.links(0)) > 0:
            cells.append(colliding_cells.links(0)[0])

        # adding correct units to displacement
        displacement_data = problem.displacement.eval([self.where], cells)

        self.data.append(displacement_data)
        self.time.append(t)

    @staticmethod
    def base_unit() -> ureg:
        """Defines the base unit of this sensor"""
        return ureg.meter
