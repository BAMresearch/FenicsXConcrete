import dolfinx as df
import numpy as np
import ufl

from fenicsxconcrete.boundary_conditions.bcs import BoundaryConditions
from fenicsxconcrete.finite_element_problem.base_material import MaterialProblem
from fenicsxconcrete.sensor_definition.base_sensor import BaseSensor


class ReactionForceSensorBottom(BaseSensor):
    """A sensor that measure the reaction force at the bottom perpendicular to the surface"""

    def __init__(self) -> None:
        self.data = []
        self.time = []

    def measure(self, problem: MaterialProblem, t: float = 1.0) -> None:
        """
        Arguments:
            problem : FEM problem object
            t : float, optional
                time of measurement for time dependent problems
        """
        # boundary condition
        bottom_surface = problem.experiment.boundary_bottom()

        v_reac = df.fem.Function(problem.V)
        bc_generator = BoundaryConditions(problem.mesh, problem.V)
        if problem.p["dim"] == 2:
            bc_generator.add_dirichlet_bc(
                df.fem.Constant(domain=problem.mesh, c=1.0),
                bottom_surface,
                1,
                "geometrical",
                1,
            )

        elif problem.p["dim"] == 3:
            bc_generator.add_dirichlet_bc(
                df.fem.Constant(domain=problem.mesh, c=1.0),
                bottom_surface,
                2,
                "geometrical",
                2,
            )

        df.fem.set_bc(v_reac.vector, bc_generator.bcs)
        computed_force = -df.fem.assemble_scalar(df.fem.form(ufl.action(problem.residual, v_reac)))

        self.data.append(computed_force)
        self.time.append(t)
