import pytest
#from fenicsxconcrete.experimental_setup.cantilever_beam import CantileverBeam
from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity
import copy

@pytest.mark.parametrize("material_model", [LinearElasticity])
def test_default_dictionaries(material_model):
    """This function creates experimental setups with the respective default dictionaries

    This makes sure all relevant values are included"""

    default_setup, default_parameters = material_model.default_parameters()

    fem_problem = material_model(default_setup, default_parameters)
    fem_problem.solve()

    # test that each parameter is truly required
    # a loop over all default parameters removes each on in turn and expects a key error from the initialized problem
    for key in default_parameters:
        with pytest.raises(KeyError):
            less_parameters = copy.deepcopy(default_parameters)
            less_parameters.pop(key)
            fem_problem = material_model(default_setup, less_parameters)
            fem_problem.solve()


