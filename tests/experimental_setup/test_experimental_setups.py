import pytest
from fenicsxconcrete.experimental_setup.cantilever_beam import CantileverBeam
from fenicsxconcrete.experimental_setup.tensile_beam import TensileBeam
import copy
from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity


@pytest.mark.parametrize("setup", [CantileverBeam,
                                   TensileBeam])
def test_default_parameters(setup):
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
            fem_problem = default_material(experiment,default_material.default_parameters()[1])
            fem_problem.solve()




