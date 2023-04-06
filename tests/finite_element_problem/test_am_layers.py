import os
from pathlib import Path

import pytest

from fenicsxconcrete.experimental_setup.am_multiple_layers import AmMultipleLayers
from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity
from fenicsxconcrete.helper import Parameters
from fenicsxconcrete.sensor_definition.reaction_force_sensor import ReactionForceSensor
from fenicsxconcrete.unit_registry import ureg


def set_test_parameters(dim: int) -> Parameters:
    """set up a test parameter set

    Args:
        dim: dimension of problem

    Returns: filled instance of Parameters

    """
    setup_parameters = Parameters()

    setup_parameters["dim"] = dim * ureg("")
    # setup_parameters["stress_state"] = "plane_strain"
    setup_parameters["num_layer"] = 5 * ureg("")  # changed in single layer test!!
    setup_parameters["layer_height"] = 1 / 100 * ureg("m")
    setup_parameters["layer_length"] = 50 / 100 * ureg("m")
    setup_parameters["layer_width"] = 5 / 100 * ureg("m")

    setup_parameters["num_elements_layer_length"] = 10 * ureg("")
    setup_parameters["num_elements_layer_height"] = 1 * ureg("")
    setup_parameters["num_elements_layer_width"] = 2 * ureg("")

    setup_parameters["rho"] = 2070.0 * ureg("kg/m^3")
    setup_parameters["E"] = 0.078e6 * ureg("N/m^2")
    setup_parameters["nu"] = 0.3 * ureg("")
    # setup_parameters['g'] = 9.81 # in material_problem.py default value

    return setup_parameters


@pytest.mark.parametrize("dimension", [2, 3])
def test_am_single_layer(dimension: int) -> None:
    """
    simple test of am experiment set up with dummy linear elastic material to get coverage
    Note: to be changed if AM MaterilapProblem is implemented"""

    # defining parameters
    setup_parameters = set_test_parameters(dimension)

    # setting up the problem
    experiment = AmMultipleLayers(setup_parameters)
    problem = LinearElasticity(experiment, setup_parameters)
    problem.add_sensor(ReactionForceSensor())

    # solving and plotting
    problem.solve()
    problem.pv_plot()

    # check sensor output
    force_bottom = problem.sensors["ReactionForceSensor"].get_last_entry().magnitude

    dead_load = (
        problem.parameters["g"]
        * problem.parameters["rho"]
        * problem.parameters["layer_length"]
        * problem.parameters["num_layer"]
        * problem.parameters["layer_height"]
    )
    if dimension == 2:
        dead_load *= 1 * ureg("m")
    elif dimension == 3:
        dead_load *= setup_parameters["layer_width"]

    # dead load of full structure
    assert force_bottom[-1] == pytest.approx(-dead_load.magnitude)
