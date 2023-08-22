from __future__ import annotations

import os
from typing import TYPE_CHECKING

import dolfinx as df
import ufl

if TYPE_CHECKING:
    from fenicsxconcrete.finite_element_problem.base_material import MaterialProblem

from fenicsxconcrete.sensor_definition.base_sensor import PointSensor
from fenicsxconcrete.util import project, ureg


class StressSensor(PointSensor):
    """A sensor that measures stress at a specific point

    Attributes:
        data: list of measured values
        time: list of time stamps
        units : pint definition of the base unit a sensor returns
        name : name of the sensor, default is class name, but can be changed
        where: location where the value is measured
    """

    def measure(self, problem: MaterialProblem) -> None:
        """
        The stress value at the defined point is added to the data list,
        as well as the time t to the time list

        Arguments:
            problem : FEM problem object
            t : time of measurement for time dependent problems, default is 1
        """
        # project stress onto visualization space
        try:
            stress = problem.q_fields.stress
            assert stress is not None
        except AssertionError:
            raise Exception("Stress not defined in problem")

        stress_function = project(
            stress,  # stress fct from problem
            df.fem.TensorFunctionSpace(problem.experiment.mesh, problem.q_fields.plot_space_type),  # tensor space
            problem.q_fields.measure,
        )

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
        stress_data = stress_function.eval([self.where], cells)

        self.data.append(stress_data)
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
        return ureg("N/m^2")
