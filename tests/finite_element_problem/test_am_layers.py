import os
from pathlib import Path

import numpy as np
import pytest

from fenicsxconcrete.experimental_setup.am_multiple_layers import AmMultipleLayers
from fenicsxconcrete.finite_element_problem.concrete_am import ConcreteAM, ConcreteThixElasticModel
from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity
from fenicsxconcrete.helper import Parameters
from fenicsxconcrete.sensor_definition.reaction_force_sensor import ReactionForceSensor
from fenicsxconcrete.sensor_definition.stress_sensor import StressSensor
from fenicsxconcrete.unit_registry import ureg


def set_test_parameters(dim: int) -> Parameters:
    """set up a test parameter set

    Args:
        dim: dimension of problem

    Returns: filled instance of Parameters

    """
    setup_parameters = {}

    setup_parameters["dim"] = dim * ureg("")
    # setup_parameters["stress_state"] = "plane_strain"
    setup_parameters["num_layers"] = 5 * ureg("")  # changed in single layer test!!
    setup_parameters["layer_height"] = 1 / 100 * ureg("m")  # y (2D), z (3D)
    setup_parameters["layer_length"] = 50 / 100 * ureg("m")  # x
    setup_parameters["layer_width"] = 5 / 100 * ureg("m")  # y (3D)

    setup_parameters["num_elements_layer_length"] = 10 * ureg("")
    setup_parameters["num_elements_layer_height"] = 1 * ureg("")
    setup_parameters["num_elements_layer_width"] = 2 * ureg("")

    if dim == 2:
        setup_parameters["stress_state"] = "plane_stress" * ureg("")

    # default material parameters as start
    _, default_params = ConcreteAM.default_parameters()
    # _, default_params = LinearElasticity.default_parameters()
    setup_parameters.update(default_params)
    if dim == 3:
        setup_parameters["q_degree"] = 4 * ureg("")

    return setup_parameters


@pytest.mark.parametrize("dimension", [2, 3])
def test_am_single_layer(dimension: int) -> None:
    """single layer test

    one layer build immediately and lying for a given time

    Args:
        dimension: dimension
    """

    # setup paths and directories
    data_dir = "data_files"
    data_path = Path(__file__).parent / data_dir

    # define file name and path for paraview output
    file_name = f"test_am_single_layer_{dimension}d"
    files = [data_path / (file_name + ".xdmf"), data_path / (file_name + ".h5")]
    # delete file if it exisits (only relevant for local tests)
    for file in files:
        if file.is_file():
            os.remove(file)

    # defining parameters
    setup_parameters = set_test_parameters(dimension)
    setup_parameters["num_layer"] = 1 * ureg("")
    setup_parameters["load_time"] = 120 * ureg("s")  # interval where load is applied linear over time

    # setting up the problem
    experiment = AmMultipleLayers(setup_parameters)

    # problem = LinearElasticity(experiment, setup_parameters)
    problem = ConcreteAM(experiment, setup_parameters, ConcreteThixElasticModel, pv_name=file_name, pv_path=data_path)
    # problem.add_sensor(ReactionForceSensor())
    problem.add_sensor(StressSensor([problem.p["layer_length"] / 2, 0, 0]))

    # solving parameters
    solve_parameters = {}
    solve_parameters["time"] = 6 * 60 * ureg("s")
    solve_parameters["dt"] = 1 * 60 * ureg("s")

    problem.set_timestep(solve_parameters["dt"])

    E_o_time = []
    t = 0.0 * ureg("s")
    while t <= solve_parameters["time"]:
        print(f"solving for t={t}")
        problem.solve(t=t)
        problem.pv_plot()
        print("computed disp", problem.displacement.x.array[:].max())

        # # store Young's modulus over time
        E_o_time.append(problem.youngsmodulus.vector.array[:].max())

        t += solve_parameters["dt"]

    print("Stress sensor", problem.sensors["StressSensor"].data)
    # print("time", problem.sensors["StressSensor"].time)
    # print("E modul", E_o_time)

    # check sensor output
    # print(np.array(problem.sensors["ReactionForceSensor"].data)[:, -1])
    # force_bottom = np.array(problem.sensors["ReactionForceSensor"].data)[:, -1]

    dead_load = (
        problem.p["g"]
        * problem.p["rho"]
        * problem.p["layer_length"]
        * problem.p["num_layers"]
        * problem.p["layer_height"]
    )
    if dimension == 2:
        dead_load *= 1  # m
    elif dimension == 3:
        dead_load *= problem.p["layer_width"]

    # dead load of full structure
    print("check", force_bottom, dead_load)
    assert force_bottom[-1] == pytest.approx(-dead_load)


if __name__ == "__main__":

    test_am_single_layer(2)
