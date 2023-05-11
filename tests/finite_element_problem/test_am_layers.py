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
from fenicsxconcrete.sensor_definition.strain_sensor import StrainSensor
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
    problem = ConcreteAM(
        experiment, setup_parameters, nonlinear_problem=ConcreteThixElasticModel, pv_name=file_name, pv_path=data_path
    )
    problem.add_sensor(ReactionForceSensor())
    problem.add_sensor(StressSensor([problem.p["layer_length"] / 2, 0, 0]))
    problem.add_sensor(StrainSensor([problem.p["layer_length"] / 2, 0, 0]))

    problem.set_timestep(solve_parameters["dt"])

    E_o_time = []
    t = 0.0 * ureg("s")
    while t <= solve_parameters["time"]:
        print(f"solving for t={t}")
        problem.solve(t=t)
        problem.pv_plot(t=t)

        # # store Young's modulus over time
        E_o_time.append(problem.youngsmodulus.vector.array[:].max())

        t += solve_parameters["dt"]

    # check reaction force
    force_bottom_y = np.array(problem.sensors["ReactionForceSensor"].data)[:, -1]
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
    print("check", sum(force_bottom_y), dead_load)
    assert sum(force_bottom_y) == pytest.approx(-dead_load)

    # check stresses change according to Emodul change
    sig_o_time = np.array(problem.sensors["StressSensor"].data)[:, -1]
    eps_o_time = np.array(problem.sensors["StrainSensor"].data)[:, -1]
    print("sig o time", sig_o_time)
    print("eps o time", eps_o_time)
    if factor == 1:
        # instance loading -> no changes
        assert sum(np.diff(sig_o_time)) == pytest.approx(0, abs=1e-8)
        assert sum(np.diff(eps_o_time)) == pytest.approx(0, abs=1e-8)
    elif factor == 2:
        # ratio sig/eps t=0 to sig/eps t=0+dt
        E_ratio_computed = (sig_o_time[0] / eps_o_time[0]) / (np.diff(sig_o_time)[0] / np.diff(eps_o_time)[0])
        assert E_ratio_computed == pytest.approx(E_o_time[0] / E_o_time[1])
        # after second time step nothing should change anymore
        assert sum(np.diff(sig_o_time)[factor - 1 : :]) == pytest.approx(0, abs=1e-8)
        assert sum(np.diff(eps_o_time)[factor - 1 : :]) == pytest.approx(0, abs=1e-8)


@pytest.mark.parametrize("dimension", [2])
@pytest.mark.parametrize("mat", ["thix"])
def test_am_multiple_layer(dimension: int, mat: str, plot: bool = False) -> None:
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
    solve_parameters["dt"] = time_layer / 4

    # defining different loading
    setup_parameters["load_time"] = 2 * solve_parameters["dt"]  # interval where load is applied linear over time

    # setting up the problem
    experiment = AmMultipleLayers(setup_parameters)
    if mat.lower() == "thix":
        problem = ConcreteAM(
            experiment,
            setup_parameters,
            nonlinear_problem=ConcreteThixElasticModel,
            pv_name=file_name,
            pv_path=data_path,
        )
    else:
        print(f"nonlinear problem {mat} not yet implemented")

    problem.set_timestep(solve_parameters["dt"])

    # initial path function describing layer activation
    path_activation = define_path(
        problem, time_layer.magnitude, t_0=-(setup_parameters["num_layers"].magnitude - 1) * time_layer.magnitude
    )
    problem.set_initial_path(path_activation)

    problem.add_sensor(ReactionForceSensor())
    problem.add_sensor(StressSensor([problem.p["layer_length"] / 2, 0, 0]))
    problem.add_sensor(StrainSensor([problem.p["layer_length"] / 2, 0, 0]))

    t = 0.0 * ureg("s")
    while t <= solve_parameters["time"]:
        print(f"solving for t={t}")
        problem.solve(t=t)
        problem.pv_plot(t=t)
        print("computed disp", problem.displacement.x.array[:].max())

        t += solve_parameters["dt"]

    # check residual force bottom
    force_bottom_y = np.array(problem.sensors["ReactionForceSensor"].data)[:, -1]
    print("force", force_bottom_y)
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

    print("check", sum(force_bottom_y), dead_load)
    assert sum(force_bottom_y) == pytest.approx(-dead_load)

    # check E modulus evolution over structure (each layer different E)
    if mat.lower() == "thix":
        if solve_parameters["time"].magnitude >= problem.p["t_f"]:
            E_bottom_layer = (
                problem.p["E_0"]
                + problem.p["R_E"] * problem.p["t_f"]
                + problem.p["A_E"] * (solve_parameters["time"].magnitude + problem.p["age_0"])
            )
            E_upper_layer = (
                problem.p["E_0"]
                + problem.p["R_E"] * problem.p["t_f"]
                + problem.p["A_E"]
                * (
                    solve_parameters["time"].magnitude
                    - (problem.p["num_layers"] - 1) * time_layer.magnitude  # layers before
                    + problem.p["age_0"]
                )
            )
        else:
            E_bottom_layer = problem.p["E_0"] + problem.p["R_E"] * (
                solve_parameters["time"].magnitude + problem.p["age_0"]
            )
            E_upper_layer = problem.p["E_0"] + problem.p["R_E"] * (
                solve_parameters["time"].magnitude
                - (problem.p["num_layers"] - 1) * time_layer.magnitude  # layers before
                + problem.p["age_0"]
            )

        print("E_bottom, E_upper", E_bottom_layer, E_upper_layer)
        print(problem.youngsmodulus.vector.array[:].min(), problem.youngsmodulus.vector.array[:].max())
        assert problem.youngsmodulus.vector.array[:].min() == pytest.approx(E_upper_layer)
        assert problem.youngsmodulus.vector.array[:].max() == pytest.approx(E_bottom_layer)
    #
    if plot:
        # example plotting
        strain_yy = np.array(problem.sensors["StrainSensor"].data)[:, -1]
        time = []
        [time.append(ti.magnitude) for ti in problem.sensors["StrainSensor"].time]

        import matplotlib.pylab as plt

        plt.figure(1)
        plt.plot(time, strain_yy, "*-r")
        plt.xlabel("process time")
        plt.ylabel("sensor bottom middle")
        plt.show()


def define_path(prob, t_diff, t_0=0):
    """create path as layer wise at quadrature space

    one layer by time

    prob: problem
    param: parameter dictionary
    t_diff: time difference between each layer
    t_0: start time for all (0 if static computation)
                            (-end_time last layer if dynamic computation)
    """

    # init path time array
    q_path = prob.rule.create_quadrature_array(prob.mesh, shape=1)

    # get quadrature coordinates with work around since tabulate_dof_coordinates()[:] not possible for quadrature spaces!
    V = df.fem.VectorFunctionSpace(prob.mesh, ("CG", prob.p["degree"]))
    v_cg = df.fem.Function(V)
    v_cg.interpolate(lambda x: (x[0], x[1]))
    positions = QuadratureEvaluator(v_cg, prob.mesh, prob.rule)
    x = positions.evaluate()
    dof_map = np.reshape(x.flatten(), [len(q_path), 2])

    # select layers only by layer height - y
    y_CO = np.array(dof_map)[:, 1]
    h_min = np.arange(0, prob.p["num_layers"] * prob.p["layer_height"], prob.p["layer_height"])
    h_max = np.arange(
        prob.p["layer_height"],
        (prob.p["num_layers"] + 1) * prob.p["layer_height"],
        prob.p["layer_height"],
    )
    # print("y_CO", y_CO)
    # print("h_min", h_min)
    # print("h_max", h_max)
    new_path = np.zeros_like(q_path)
    EPS = 1e-8
    for i in range(0, len(h_min)):
        layer_index = np.where((y_CO > h_min[i] - EPS) & (y_CO <= h_max[i] + EPS))
        new_path[layer_index] = t_0 + (prob.p["num_layers"] - 1 - i) * t_diff

    q_path = new_path

    return q_path


if __name__ == "__main__":

    # test_am_single_layer(2, 2)

    test_am_multiple_layer(2, "thix", True)
