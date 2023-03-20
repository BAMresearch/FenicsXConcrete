import dolfinx as df
import numpy as np
from fenicsxconcrete.unit_registry import ureg
from fenicsxconcrete.sensor_definition.base_sensor import Sensor

class DisplacementSensor(Sensor):
    """A sensor that measure displacement at a specific point"""

    def __init__(self, where):
        """
        Arguments:
            where : Point
                location where the value is measured
        """
        self.where = where
        self.data = []
        self.time = []

    def measure(self, problem, t=1.0):
        """
        Arguments:
            problem : FEM problem object
            t : float, optional
                time of measurement for time dependent problems
        """
        # get displacements
        bb_tree = df.geometry.BoundingBoxTree(problem.experiment.mesh, problem.experiment.mesh.topology.dim)
        cells = []

        # Find cells whose bounding-box collide with the the points
        cell_candidates = df.geometry.compute_collisions(bb_tree, self.where)

        # Choose one of the cells that contains the point
        colliding_cells = df.geometry.compute_colliding_cells(problem.experiment.mesh, cell_candidates, self.where)

        for i, point in enumerate(self.where):
            if len(colliding_cells.links(i))>0:
                cells.append(colliding_cells.links(i)[0])

        # adding correct units to displacement
        # TODO: is this the best/correct way to add the units?
        #       should the list have units, or should each element have a unit?
        #       it would be better to be able to define "base_length_unit" instead of "meter"
        displacement_data = problem.displacement.eval(self.where, cells) * ureg('m')

        self.data.append(displacement_data)
        self.time.append(t)