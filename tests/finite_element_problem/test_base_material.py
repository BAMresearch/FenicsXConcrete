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

    default_experiment, fem_parameters = LinearElasticity.default_parameters()
    fem_parameters['log_level'] = log_str * ureg('')

    if not error_flag:
        LinearElasticity(default_experiment, fem_parameters)
    else:
        with pytest.raises(ValueError):
            LinearElasticity(default_experiment, fem_parameters)


def test_sensor_error():
    """This function tests the add sensor function"""

    default_experiment, fem_parameters = LinearElasticity.default_parameters()
    problem = LinearElasticity(default_experiment, fem_parameters)

    with pytest.raises(ValueError):
        problem.add_sensor('not a sensor')