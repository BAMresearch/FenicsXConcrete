import pytest
from fenicsxconcrete.experimental_setup.cantilever_beam import CantileverBeam
from fenicsxconcrete.experimental_setup.tensile_beam import TensileBeam


@pytest.mark.parametrize("setup", [CantileverBeam,
                                   TensileBeam])
def test_default_parameters(setup):
    """This function creates experimental setups with the respective default dictionaries

    This makes sure all relevant values are included"""

    setup_parameters = setup.default_parameters()

    # initialize with default parameters
    experiment = setup(setup_parameters)

    # test that each parameter is truly required
    for key in setup_parameters:
        with pytest.raises(KeyError):
            less_parameters = setup.default_parameters()
            less_parameters.pop(key)
            experiment = setup(less_parameters)

