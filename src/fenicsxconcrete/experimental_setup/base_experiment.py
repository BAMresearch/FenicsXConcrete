from abc import ABC, abstractmethod

import dolfinx as df
import pint
import ufl

from fenicsxconcrete.helper import LogMixin, Parameters
from fenicsxconcrete.unit_registry import ureg


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
        """define empty displacement boundary conditions (to be done in child)

        this function is abstract until there is a need for a material that does not need a displacement boundary
        once that is required, just make this a normal function that returns an empty list

        Args:
            V: function space

        Returns:
            if defined a list with displacement boundary conditions otherwise None

        """

    def create_force_boundary(self, v: ufl.argument.Argument | None = None) -> ufl.form.Form | None:
        """define empty force boundary (to be done in child)

        Args:
            v: test function

        Returns:
            if defined a form for the force otherwise None

        """

        pass

    def create_body_force(self, v: ufl.argument.Argument | None = None) -> ufl.form.Form | None:
        """define empty body force function

        Args:
            v: test function

        Returns:
            if defined a form for the body force otherwise None

        """

        pass
