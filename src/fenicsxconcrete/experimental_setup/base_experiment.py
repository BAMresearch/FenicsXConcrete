import dolfinx as df
import numpy as np
from fenicsxconcrete.helper import Parameters
from abc import ABC, abstractmethod
import pint


class Experiment(ABC):
    """base class for experimental setups

    Attributes:
        parameters : parameter dictionary with units
        p : parameter dictionary without units

    """

    def __init__(self, parameters: dict[str, pint.Quantity]):
        """Initialises the parent object

        This is needs to be called by children
        Constant parameters are defined here
        """

        # initialize parameter attributes
        self.parameters = parameters
        self.p = self.parameters.to_magnitude()

        self.setup()

    @abstractmethod
    def setup(self):
        """Is called by init, must be defined by child"""
        pass

    @staticmethod
    @abstractmethod
    def default_parameters() -> dict[str, pint.Quantity]:
        """returns a dictionary with required parameters and a set of working values as example"""
        # this must de defined in each setup class
        pass

    def create_displacement_boundary(self, V) -> list:
        # define empty displacement boundary
        displ_bcs = []

        return displ_bcs

    def create_force_boundary(self, v):
        # define empty force boundary
        # TODO: is there a better solution???

        return None

    def create_body_force(self, v):
        # define empty body force function
        # TODO: is there a better solution???

        return None