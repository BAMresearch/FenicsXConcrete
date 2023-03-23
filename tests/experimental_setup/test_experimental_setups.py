import copy

import pytest

from fenicsxconcrete.experimental_setup.am_multiple_layers import AmMultipleLayers
from fenicsxconcrete.experimental_setup.base_experiment import Experiment
from fenicsxconcrete.experimental_setup.cantilever_beam import CantileverBeam
from fenicsxconcrete.experimental_setup.compression_cylinder import CompressionCylinder
from fenicsxconcrete.experimental_setup.simple_beam import SimpleBeam
from fenicsxconcrete.experimental_setup.tensile_beam import TensileBeam
from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity
from fenicsxconcrete.unit_registry import ureg


@pytest.mark.parametrize("setup", [CantileverBeam, TensileBeam, SimpleBeam, CompressionCylinder, AmMultipleLayers])
def test_default_parameters(setup: Experiment) -> None:
    """This function creates experimental setups with the respective default dictionaries

    This makes sure all relevant values are included"""
    default_material = LinearElasticity

    setup_parameters = setup.default_parameters()

    # initialize with default parameters
    experiment = setup(setup_parameters)

    # test that each parameter is truly required
    for key in setup_parameters:
        with pytest.raises(KeyError):
            less_parameters = copy.deepcopy(setup_parameters)
            less_parameters.pop(key)
            experiment = setup(less_parameters)
            fem_problem = default_material(experiment, default_material.default_parameters()[1])
            fem_problem.solve()


# to imporve coverage, I want to test the error messages
@pytest.mark.parametrize("setup", [CantileverBeam, TensileBeam, SimpleBeam, CompressionCylinder, AmMultipleLayers])
def test_default_parameters(setup: Experiment) -> None:
    setup_parameters = setup.default_parameters()

    with pytest.raises(ValueError):
        setup_parameters["dim"] = 4 * ureg("")  # there is no 4D setup
        test_setup = setup(setup_parameters)
