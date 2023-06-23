from abc import ABC, abstractmethod
from collections.abc import Callable

import dolfinx as df
import pint
import ufl

from fenicsxconcrete.boundary_conditions.boundary import plane_at, point_at
from fenicsxconcrete.util import LogMixin, Parameters, QuadratureRule, ureg


class Experiment(ABC, LogMixin):
    """base class for experimental setups

    Attributes:
        parameters : parameter dictionary with units
        p : parameter dictionary without units

    """

    def __init__(self, parameters: dict[str, pint.Quantity]) -> None:
        """Initialises the parent object

        This is needs to be called by children
        Constant parameters are defined here

        Args:
            parameters: parameter dictionary with units

        """

        # initialize parameter attributes
        default_setup_parameters = Parameters()
        # setting up default setup parameters
        default_setup_parameters["degree"] = 2 * ureg("")  # polynomial degree

        # update with input parameters
        default_setup_parameters.update(parameters)
        # as attribute
        self.parameters = default_setup_parameters
        # remove units for use in fem model
        self.p = self.parameters.to_magnitude()

        self.setup()

    @abstractmethod
    def setup(self):
        """Is called by init, must be defined by child"""

    @staticmethod
    @abstractmethod
    def default_parameters() -> dict[str, pint.Quantity]:
        """sets up a working set of parameter values as example

        must be defined in each child

        Returns:
            a dictionary with required parameters and a set of working values as example

        """

    @abstractmethod
    def create_displacement_boundary(self, V: df.fem.FunctionSpace) -> list[df.fem.bcs.DirichletBCMetaClass] | None:
        """defines empty displacement boundary conditions (to be done in child)

        this function is abstract until there is a need for a material that does not need a displacement boundary
        once that is required, just make this a normal function that returns an empty list

        Args:
            V: function space

        Returns:
            if defined a list with displacement boundary conditions otherwise None

        """

    def create_force_boundary(self, v: ufl.argument.Argument | None = None) -> ufl.form.Form | None:
        """defines empty force boundary (to be done in child)

        Args:
            v: test function

        Returns:
            if defined a form for the force otherwise None

        """

        pass

    def create_body_force(self, v: ufl.argument.Argument | None = None) -> ufl.form.Form | None:
        """defines empty body force function

        Args:
            v: test function

        Returns:
            if defined a form for the body force otherwise None

        """

        pass

    def create_body_force_am(
        self,
        v: ufl.argument.Argument | None = None,
        q_fd: df.fem.Function | None = None,
        rule: QuadratureRule | None = None,
    ) -> ufl.form.Form | None:
        """defines empty body force function for am case

        Args:
            v: test function
            q_fd: quadrature function given the loading increment where elements are active
            rule: rule for the quadrature function

        Returns:
            if defined a form for the body force otherwise None

        """

        pass

    def boundary_top(self) -> Callable:
        """specifies boundary: plane at top

        Returns:
            fct defining if dof is at boundary

        """
        if self.p["dim"] == 2:
            return plane_at(self.p["height"], 1)
        elif self.p["dim"] == 3:
            return plane_at(self.p["height"], 2)

    def boundary_bottom(self) -> Callable:
        """specifies boundary: plane at bottom

        Returns: fct defining if dof is at boundary

        """
        if self.p["dim"] == 2:
            return plane_at(0.0, "y")
        elif self.p["dim"] == 3:
            return plane_at(0.0, "z")

    def boundary_left(self) -> Callable:
        """specifies boundary: plane at left side

        Returns:
            fct defining if dof is at boundary

        """
        if self.p["dim"] == 2:
            return plane_at(0.0, "x")
        elif self.p["dim"] == 3:
            return plane_at(0.0, "x")

    def boundary_right(self) -> Callable:
        """specifies boundary: plane at left side

        Returns:
            fct defining if dof is at boundary

        """
        if self.p["dim"] == 2:
            return plane_at(self.p["length"], "x")
        elif self.p["dim"] == 3:
            return plane_at(self.p["length"], "x")

    def boundary_front(self) -> Callable:
        """specifies boundary: plane at front

        only for 3D case front plane

        Returns:
            fct defining if dof is at boundary

        """
        if self.p["dim"] == 3:
            return plane_at(0.0, "y")

    def boundary_back(self) -> Callable:
        """specifies boundary: plane at front

        only for 3D case front plane

        Returns:
            fct defining if dof is at boundary

        """
        if self.p["dim"] == 3:
            return plane_at(self.p["width"], "y")
