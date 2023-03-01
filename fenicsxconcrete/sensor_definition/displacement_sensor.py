import dolfinx as df
import numpy as np
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
        #self.data.append(problem.displacement(self.where))

        bb_tree = df.geometry.BoundingBoxTree(problem.experiment.mesh, problem.experiment.mesh.topology.dim)
        cells = []
        points_on_proc = []

        # Find cells whose bounding-box collide with the the points
        cell_candidates = df.geometry.compute_collisions(bb_tree, self.where)

        # Choose one of the cells that contains the point
        colliding_cells = df.geometry.compute_colliding_cells(problem.experiment.mesh, cell_candidates, self.where)
        for i, point in enumerate(self.where):
            if len(colliding_cells.links(i))>0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])
        points_on_proc = np.array(points_on_proc, dtype=np.float64)
        self.data.append(problem.displacement.eval(points_on_proc, cells))
        self.time.append(t)