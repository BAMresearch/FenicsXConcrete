from __future__ import annotations

import os
from typing import TYPE_CHECKING

import dolfinx as df

if TYPE_CHECKING:
    from fenicsxconcrete.finite_element_problem.base_material import MaterialProblem

from fenicsxconcrete.sensor_definition.base_sensor import PointSensor
from fenicsxconcrete.util import ureg


class DisplacementSensor(PointSensor):
    """A sensor that measures displacement at a specific point

    Attributes:
        data: list of measured values
        time: list of time stamps
        units : pint definition of the base unit a sensor returns
        name : name of the sensor, default is class name, but can be changed
        where: location where the value is measured
    """

    # Type hints don't work here because they create a circular import :(
    def measure(self, problem: MaterialProblem) -> None:
        """
        The displacement value at the defined point is added to the data list,
        as well as the time t to the time list

        Arguments:
            problem : FEM problem object
            t : time of measurement for time dependent problems, default is 1
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
        displacement_data = problem.fields.displacement.eval([self.where], cells)

        self.data.append(displacement_data)
        self.time.append(problem.time)

    def report_metadata(self) -> dict:
        """Generates dictionary with the metadata of this sensor"""
        metadata = super().report_metadata()
        metadata["sensor_file"] = os.path.splitext(os.path.basename(__file__))[0]
        return metadata

    @staticmethod
    def base_unit() -> ureg:
        """Defines the base unit of this sensor

        Returns:
            the base unit as pint unit object
        """
        return ureg.meter
