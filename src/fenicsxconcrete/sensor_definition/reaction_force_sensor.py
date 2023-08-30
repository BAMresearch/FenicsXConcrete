from __future__ import annotations

import os
from typing import TYPE_CHECKING, TypedDict

import dolfinx as df
import ufl

if TYPE_CHECKING:
    from fenicsxconcrete.finite_element_problem.base_material import MaterialProblem

import dolfinx as df
import ufl

from fenicsxconcrete.boundary_conditions.bcs import BoundaryConditions
from fenicsxconcrete.sensor_definition.base_sensor import BaseSensor
from fenicsxconcrete.util import ureg


class Surface(TypedDict):
    """A typed dictionary class to define a surface

    Attributes:
        function: name of the function to be called to ge the surface
        args: additional arguments to be passed to such a function
    """

    function: str
    args: dict


class ReactionForceSensor(BaseSensor):
    """A sensor that measures the reaction force at a specified surface

    Attributes:
        data: list of measured values
        time: list of time stamps
        units : pint definition of the base unit a sensor returns
        name : name of the sensor, default is class name, but can be changed
        surface : dictionary that defines the surface where the reaction force is measured
    Args:
        surface : a dictionary that defines the function for the reaction boundary, default is the bottom surface
        name : name of the sensor, default is class name, but can be changed
    """

    def __init__(self, surface: Surface | None = None, name: str | None = None) -> None:
        super().__init__(name=name)
        self.surface_dict = surface

    def measure(self, problem: MaterialProblem) -> None:
        """
        The reaction force vector of the defined surface is added to the data list,
        as well as the time t to the time list

        Args:
            problem : FEM problem object
            t : time of measurement for time dependent problems, default is 1
        """
        # boundary condition
        if self.surface_dict is None:
            self.surface = problem.experiment.boundary_bottom()
        else:
            self.surface = getattr(problem.experiment, self.surface_dict["function"])(**self.surface_dict["args"])

        v_reac = df.fem.Function(problem.fields.displacement.function_space)

        reaction_force_vector = []

        bc_generator_x = BoundaryConditions(problem.mesh, problem.V)
        bc_generator_x.add_dirichlet_bc(
            value=df.fem.Constant(domain=problem.mesh, c=1.0),
            boundary=self.surface,
            sub=0,
            method="geometrical",
            entity_dim=problem.mesh.topology.dim - 1,
        )
        df.fem.set_bc(v_reac.vector, bc_generator_x.bcs)
        computed_force_x = -df.fem.assemble_scalar(df.fem.form(ufl.action(problem.residual, v_reac)))
        reaction_force_vector.append(computed_force_x)

        bc_generator_y = BoundaryConditions(problem.mesh, problem.V)
        bc_generator_y.add_dirichlet_bc(
            value=df.fem.Constant(domain=problem.mesh, c=1.0),
            boundary=self.surface,
            sub=1,
            method="geometrical",
            entity_dim=problem.mesh.topology.dim - 1,
        )
        df.fem.set_bc(v_reac.vector, bc_generator_y.bcs)
        computed_force_y = -df.fem.assemble_scalar(df.fem.form(ufl.action(problem.residual, v_reac)))
        reaction_force_vector.append(computed_force_y)

        if problem.p["dim"] == 3:
            bc_generator_z = BoundaryConditions(problem.mesh, problem.V)
            bc_generator_z.add_dirichlet_bc(
                value=df.fem.Constant(domain=problem.mesh, c=1.0),
                boundary=self.surface,
                sub=2,
                method="geometrical",
                entity_dim=problem.mesh.topology.dim - 1,
            )
            df.fem.set_bc(v_reac.vector, bc_generator_z.bcs)
            computed_force_z = -df.fem.assemble_scalar(df.fem.form(ufl.action(problem.residual, v_reac)))
            reaction_force_vector.append(computed_force_z)

        self.data.append(reaction_force_vector)
        self.time.append(problem.time)

    def report_metadata(self) -> dict:
        """Generates dictionary with the metadata of this sensor"""
        metadata = super().report_metadata()
        metadata["surface"] = self.surface_dict
        metadata["sensor_file"] = os.path.splitext(os.path.basename(__file__))[0]
        return metadata

    @staticmethod
    def base_unit() -> ureg:
        """Defines the base unit of this sensor

        Returns:
            the base unit as pint unit object
        """
        return ureg.newton
