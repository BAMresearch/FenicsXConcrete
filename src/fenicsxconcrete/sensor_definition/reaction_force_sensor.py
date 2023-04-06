import dolfinx as df
import numpy as np
import ufl

from fenicsxconcrete.boundary_conditions.bcs import BoundaryConditions
from fenicsxconcrete.finite_element_problem.base_material import MaterialProblem
from fenicsxconcrete.sensor_definition.base_sensor import BaseSensor
from fenicsxconcrete.unit_registry import ureg


class ReactionForceSensor(BaseSensor):
    """A sensor that measure the reaction force at the bottom perpendicular to the surface"""

    def __init__(self, surface=None) -> None:
        """
        Arguments:
            where : Point where to measure
        """
        super().__init__()
        self.surface = surface

    def measure(self, problem: MaterialProblem, t: float = 1.0) -> None:
        """
        Arguments:
            problem : FEM problem object
            t : float, optional
                time of measurement for time dependent problems
        """
        # boundary condition
        if self.surface is None:
            self.surface = problem.experiment.boundary_bottom()

        v_reac = df.fem.Function(problem.V)

        reaction_force_vector = []

        bc_generator_x = BoundaryConditions(problem.mesh, problem.V)
        bc_generator_x.add_dirichlet_bc(df.fem.Constant(domain=problem.mesh, c=1.0), self.surface, 0, "geometrical", 0)
        df.fem.set_bc(v_reac.vector, bc_generator_x.bcs)
        computed_force_x = -df.fem.assemble_scalar(df.fem.form(ufl.action(problem.residual, v_reac)))
        reaction_force_vector.append(computed_force_x)

        bc_generator_y = BoundaryConditions(problem.mesh, problem.V)
        bc_generator_y.add_dirichlet_bc(df.fem.Constant(domain=problem.mesh, c=1.0), self.surface, 1, "geometrical", 1)
        df.fem.set_bc(v_reac.vector, bc_generator_y.bcs)
        computed_force_y = -df.fem.assemble_scalar(df.fem.form(ufl.action(problem.residual, v_reac)))
        reaction_force_vector.append(computed_force_y)

        if problem.p["dim"] == 3:
            bc_generator_z = BoundaryConditions(problem.mesh, problem.V)
            bc_generator_z.add_dirichlet_bc(
                df.fem.Constant(domain=problem.mesh, c=1.0), self.surface, 2, "geometrical", 2
            )
            df.fem.set_bc(v_reac.vector, bc_generator_z.bcs)
            computed_force_z = -df.fem.assemble_scalar(df.fem.form(ufl.action(problem.residual, v_reac)))
            reaction_force_vector.append(computed_force_z)

        self.data.append(reaction_force_vector)
        self.time.append(t)

    @staticmethod
    def base_unit() -> ureg:
        """Defines the base unit of this sensor"""
        return ureg.newton
