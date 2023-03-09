import pytest
from fenicsxconcrete.experimental_setup.cantilever_beam import CantileverBeam
from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity
from fenicsxconcrete.unit_registry import ureg



@pytest.mark.parametrize("log_level", [['DEBUG',False],
                                       ['INFO',False],
                                       ['ERROR',False],
                                       ['WARNING',False],
                                       ['CRITICAL',False],
                                       ['some_string_that is_not_implemented',True]
                                       ])
def test_log_levels(log_level):
    """This function tests all implemented log level in the base material init"""

    log_str = log_level[0]
    error_flag = log_level[1]

    setup_parameters = CantileverBeam.default_parameters()
    setup_parameters['log_level'] = log_str * ureg('')
    experiment = CantileverBeam(setup_parameters)
    default_setup, fem_parameters = LinearElasticity.default_parameters()

    if not error_flag:
        LinearElasticity(experiment, fem_parameters)
    else:
        with pytest.raises(ValueError):
            LinearElasticity(experiment, fem_parameters)
