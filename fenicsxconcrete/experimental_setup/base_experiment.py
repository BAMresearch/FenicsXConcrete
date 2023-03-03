import dolfinx as df
import numpy as np
from fenicsxconcrete.helper import Parameters
from abc import ABC, abstractmethod


class Experiment:
    """base class for experimental setups

    Attributes:
        parameters : parameter dictionary with units
        p : parameter dictionary without units

    """

    def __init__(self, parameters):
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

    def create_displacement_boundary(self, V):
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