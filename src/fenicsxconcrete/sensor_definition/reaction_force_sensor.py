from collections.abc import Callable

import dolfinx as df
import ufl

from fenicsxconcrete.boundary_conditions.bcs import BoundaryConditions
from fenicsxconcrete.finite_element_problem.base_material import MaterialProblem
from fenicsxconcrete.sensor_definition.base_sensor import BaseSensor
from fenicsxconcrete.unit_registry import ureg


class ReactionForceSensor(BaseSensor):
    """A sensor that measures the reaction force at a specified surface

    Attributes:
        data: list of measured values
        time: list of time stamps
        units : pint definition of the base unit a sensor returns
        name : name of the sensor, default is class name, but can be changed
        surface : function that defines the surface where the reaction force is measured
    """

    def __init__(self, surface: Callable | None = None, name: str | None = None) -> None:
        """
        initializes a reaction force sensor, for further details, see base class

        Arguments:
            surface : a function that defines the reaction boundary, default is the bottom surface
            name : name of the sensor, default is class name, but can be changed
        """
        super().__init__(name=name)
        self.surface = surface

    def measure(self, problem: MaterialProblem, t: float = 1.0) -> None:
        """
        The reaction force vector of the defined surface is added to the data list,
        as well as the time t to the time list

        Arguments:
            problem : FEM problem object
            t : time of measurement for time dependent problems, default is 1
        """
        # boundary condition
        if self.surface is None:
            self.surface = problem.experiment.boundary_bottom()

        v_reac = df.fem.Function(problem.V)

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
        self.time.append(t)

    @staticmethod
    def base_unit() -> ureg:
        """Defines the base unit of this sensor

        Returns:
            the base unit as pint unit object
        """
        return ureg.newton
