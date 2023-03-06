import pytest
from fenicsxconcrete.experimental_setup.cantilever_beam import CantileverBeam
from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity


@pytest.mark.parametrize("material_model", [LinearElasticity])
def test_default_dictionaries(material_model):
    """This function creates experimental setups with the respective default dictionaries

    This makes sure all relevant values are included"""

    # make sure the material_model can be initialized with default values
    experiment = CantileverBeam(CantileverBeam.default_parameters())
    fem_parameters = material_model.default_parameters()
    fem_problem = material_model(experiment,material_model.default_parameters())

    # test that each parameter is truly required
    # a loop over all default parameters removes each on in turn and expects a key error from the initialized problem
    for key in fem_parameters:
        with pytest.raises(KeyError):
            less_parameters = material_model.default_parameters()
            less_parameters.pop(key)
            fem_problem = material_model(experiment, less_parameters)


