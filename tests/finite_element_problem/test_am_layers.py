import os
from pathlib import Path

import dolfinx as df
import numpy as np
import pytest

from fenicsxconcrete.experimental_setup.am_multiple_layers import AmMultipleLayers
from fenicsxconcrete.finite_element_problem.concrete_am import ConcreteAM, ConcreteThixElasticModel
from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity
from fenicsxconcrete.helper import Parameters, QuadratureEvaluator
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
    setup_parameters.update(default_params)
    if dim == 3:
        setup_parameters["q_degree"] = 4 * ureg("")

    return setup_parameters


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("factor", [1, 2])
def test_am_single_layer(dimension: int, factor: int) -> None:
    """single layer test

    one layer build immediately and lying for a given time

    Args:
        dimension: dimension
        factor: length of load_time = factor * dt
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
    setup_parameters["num_layers"] = 1 * ureg("")

    # solving parameters
    solve_parameters = {}
    solve_parameters["time"] = 6 * 60 * ureg("s")
    solve_parameters["dt"] = 60 * ureg("s")

    # defining different loading
    setup_parameters["load_time"] = factor * solve_parameters["dt"]  # interval where load is applied linear over time

    # setting up the problem
    experiment = AmMultipleLayers(setup_parameters)

    # problem = LinearElasticity(experiment, setup_parameters)
    problem = ConcreteAM(experiment, setup_parameters, ConcreteThixElasticModel, pv_name=file_name, pv_path=data_path)
    problem.add_sensor(ReactionForceSensor())
    problem.add_sensor(StressSensor([problem.p["layer_length"] / 2, 0, 0]))

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


@pytest.mark.parametrize("dimension", [2])
@pytest.mark.parametrize("mat", ["thix"])
def test_am_multiple_layer(dimension: int, mat: str) -> None:
    """multiple layer test

    several layers building over time one layer at once

    Args:
        dimension: dimension

    """

    # setup paths and directories
    data_dir = "data_files"
    data_path = Path(__file__).parent / data_dir

    # define file name and path for paraview output
    file_name = f"test_am_multiple_layer_{dimension}d"
    files = [data_path / (file_name + ".xdmf"), data_path / (file_name + ".h5")]
    # delete file if it exisits (only relevant for local tests)
    for file in files:
        if file.is_file():
            os.remove(file)

    # defining parameters
    setup_parameters = set_test_parameters(dimension)

    # solving parameters
    solve_parameters = {}
    time_layer = 20 * ureg("s")  # time to build one layer
    solve_parameters["time"] = setup_parameters["num_layers"] * time_layer
    solve_parameters["dt"] = time_layer / 2

    # defining different loading
    setup_parameters["load_time"] = 1 * solve_parameters["dt"]  # interval where load is applied linear over time

    # setting up the problem
    experiment = AmMultipleLayers(setup_parameters)
    if mat.lower() == "thix":
        problem = ConcreteAM(
            experiment, setup_parameters, ConcreteThixElasticModel, pv_name=file_name, pv_path=data_path
        )
    else:
        print(f"nonlinear problem {mat} not yet implemented")

    problem.set_timestep(solve_parameters["dt"])

    # initial path function describing layer activation
    path_activation = define_path(
        problem, time_layer.magnitude, t_0=-(setup_parameters["num_layers"].magnitude - 1) * time_layer.magnitude
    )
    problem.set_initial_path(path_activation)

    # problem.add_sensor(ReactionForceSensor())
    problem.add_sensor(StressSensor([problem.p["layer_length"] / 2, 0, 0]))

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


def define_path(prob, t_diff, t_0=0):
    """create path as layer wise at quadrature space

    one layer by time

    prob: problem
    param: parameter dictionary
    t_diff: time difference between each layer
    t_0: start time for all (0 if static computation)
                            (-end_time last layer if dynamic computation)
    """

    # # get quadrature function space
    # q_V = prob.rule.create_quadrature_space(prob.experiment.mesh)
    # # print(prob.rule.points)
    # # print(prob.rule.weights)
    # # print(q_V.tabulate_dof_coordinates())
    # print(dir(q_V))
    # # print(dir(q_V.tabulate_dof_coordinates))
    # V = df.fem.VectorFunctionSpace(prob.experiment.mesh, ("CG", 2))
    # # print(V.tabulate_dof_coordinates()[:])
    # v_cg = df.fem.Function(V)
    # v_cg.interpolate(lambda x: (x[0], x[1]))
    # positions = QuadratureEvaluator(v_cg, prob.experiment.mesh, prob.rule)
    # x = positions.evaluate()
    # print(x)
    # print(len(x))
    #
    # input()

    # standard CG function
    V = df.fem.FunctionSpace(prob.experiment.mesh, ("CG", prob.p["degree"]))
    v_cg = df.fem.Function(V)
    # dof map for coordinates
    dof_map = V.tabulate_dof_coordinates()[:]
    new_path = np.zeros(len(v_cg.vector.array[:]))

    y_CO = np.array(dof_map)[:, 1]
    h_min = np.arange(0, prob.p["num_layers"] * prob.p["layer_height"], prob.p["layer_height"])
    h_max = np.arange(
        prob.p["layer_height"],
        (prob.p["num_layers"] + 1) * prob.p["layer_height"],
        prob.p["layer_height"],
    )
    print("y_CO", y_CO)
    print("h_min", h_min)
    print("h_max", h_max)
    EPS = 1e-8
    for i in range(0, len(h_min)):
        layer_index = np.where((y_CO > h_min[i] - EPS) & (y_CO <= h_max[i] + EPS))
        # print((parameters['layer_number']-i-1)*age_diff_layer)
        new_path[layer_index] = t_0 + (prob.p["num_layers"] - 1 - i) * t_diff

    print("new_path", new_path, new_path.min(), new_path.max())
    print(len(new_path))

    v_cg.vector.array[:] = new_path
    v_cg.x.scatter_forward()

    # interpolate on quadrature space
    q_path = np.zeros_like(prob.mechanics_problem.q_array_path)
    quad_ev = QuadratureEvaluator(v_cg, prob.experiment.mesh, prob.rule)
    quad_ev.evaluate(q_path)
    print(q_path, len(q_path))

    return q_path


if __name__ == "__main__":

    test_am_single_layer(2, 2)

    # test_am_multiple_layer(2, "thix")
