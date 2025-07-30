import os
from pathlib import Path
from typing import Literal

import dolfinx as df
import numpy as np
import pytest

# for know copy material law from fenics-constitutive to tests/finite_element_problem should be a module later
from fenics_constitutive.models import LinearElasticityModel, SpringKelvinModel, SpringMaxwellModel, VonMises3D

from fenicsxconcrete.experimental_setup import AmMultipleLayers
from fenicsxconcrete.finite_element_problem.concrete_am_fc import ConcreteAMFC
from fenicsxconcrete.sensor_definition.displacement_sensor import DisplacementSensor
from fenicsxconcrete.sensor_definition.reaction_force_sensor import ReactionForceSensor
from fenicsxconcrete.sensor_definition.stress_sensor import StressSensor
from fenicsxconcrete.util import Parameters, QuadratureEvaluator, ureg


def set_test_parameters(mat: Literal["linear_elastic", "mises"]) -> Parameters:
    """set up a test parameter set

    Args:
        dim: dimension of problem

    Returns: filled instance of Parameters

    """
    setup_parameters = {}

    setup_parameters["dim"] = 3 * ureg("")
    # setup_parameters["stress_state"] = "plane_strain"
    setup_parameters["num_layers"] = 10 * ureg("")  # changed in single layer test!!
    setup_parameters["layer_height"] = 1 / 100 * ureg("m")  # y (2D), z (3D)
    setup_parameters["layer_length"] = 50 / 100 * ureg("m")  # x
    setup_parameters["layer_width"] = 3 / 100 * ureg("m")  # y (3D)

    setup_parameters["num_elements_layer_length"] = 10 * ureg("")
    setup_parameters["num_elements_layer_height"] = 1 * ureg("")
    setup_parameters["num_elements_layer_width"] = 2 * ureg("")

    setup_parameters["q_degree"] = 4 * ureg("")
    setup_parameters["rho"] = 2000 * ureg("kg/m^3")

    if mat == "linear_elastic":
        material_law = LinearElasticityModel
        setup_parameters["E"] = 42000 * ureg("Pa")  # young's modulus
        setup_parameters["nu"] = 0.3 * ureg("")  # poisson ratio
        setup_parameters["A_E"] = 4000 * ureg("Pa/s")  # young's modulus rate over time
        setup_parameters["time_fct"] = "linear" * ureg("")  # time dependency of material parameters
    elif mat == "visco_Kelvin":
        material_law = SpringKelvinModel
        setup_parameters["E0"] = 550 * ureg("Pa")
        setup_parameters["E1"] = 190 * ureg("Pa")
        setup_parameters["tau"] = 10 * ureg("s")
        setup_parameters["time_fct"] = "linear" * ureg("")  # time dependency of material parameters
        setup_parameters["A_E0"] = 10 * ureg("Pa/s")  # young's modulus rate over time
        setup_parameters["A_E1"] = 5 * ureg("Pa/s")  # young's modulus rate over time
        setup_parameters["A_tau"] = 1 * ureg("Pa/s")  # young's modulus rate over time
    elif mat == "visco_Maxwell":
        material_law = SpringMaxwellModel
        setup_parameters["E0"] = 550 * ureg("Pa")
        setup_parameters["E1"] = 190 * ureg("Pa")
        setup_parameters["tau"] = 10 * ureg("s")
        setup_parameters["time_fct"] = "linear" * ureg("")  # time dependency of material parameters
        setup_parameters["A_E0"] = 10 * ureg("Pa/s")  # young's modulus rate over time
        setup_parameters["A_E1"] = 5 * ureg("Pa/s")  # young's modulus rate over time
        setup_parameters["A_tau"] = 1 * ureg("Pa/s")  # young's modulus rate over time
    elif mat == "mises":
        material_law = VonMises3D
        # setup_parameters["p_ka"] = 17500.0 * ureg("Pa")  # bulk modulus
        # setup_parameters["p_mu"] = 8076.9 * ureg("Pa")  # shear modulus
        # setup_parameters["p_y0"] = 300 * ureg("Pa")  # initial yield stress
        # setup_parameters["p_y00"] = 2500 * ureg("Pa")  # final yield stress
        # setup_parameters["p_w"] = 200 * ureg("")  # saturation parameter

        # setup_parameters["p_ka"] = 1750.00 * ureg("Pa")  # bulk modulus
        # setup_parameters["p_mu"] = 807.69 * ureg("Pa")  # shear modulus
        # setup_parameters["p_y0"] = 20 * ureg("Pa")  # initial yield stress
        # setup_parameters["p_y00"] = 2000 * ureg("Pa")  # final yield stress
        # setup_parameters["p_w"] = 200 * ureg("")  # saturation parameter

        setup_parameters["p_ka"] = 17500.0 * ureg("Pa")  # bulk modulus
        setup_parameters["p_mu"] = 8076.9 * ureg("Pa")  # shear modulus
        setup_parameters["p_y0"] = 1000 * ureg("Pa")  # initial yield stress
        setup_parameters["p_y00"] = 2000 * ureg("Pa")  # final yield stress
        setup_parameters["p_w"] = 0.3 * ureg("")  # saturation parameter
    else:
        raise ValueError("material not supported")

    return setup_parameters, material_law


@pytest.mark.parametrize("mat", ["linear_elastic", "visco_Kelvin", "visco_Maxwell", "mises"])
@pytest.mark.parametrize("factor", [1, 2])
def test_am_single_layer(
    mat: Literal["linear_elastic", "visco_Kelvin", "visco_Maxwell", "mises"], factor: int
) -> None:
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
    file_name = "test_am_fc_single_layer"
    files = [data_path / (file_name + ".xdmf"), data_path / (file_name + ".h5")]
    # delete file if it exisits (only relevant for local tests)
    for file in files:
        if file.is_file():
            os.remove(file)

    # defining parameters
    setup_parameters, material_law = set_test_parameters(mat)
    setup_parameters["num_layers"] = 1 * ureg("")

    # defining different loading
    setup_parameters["dt"] = 6 * ureg("s")
    setup_parameters["load_time"] = factor * setup_parameters["dt"]  # interval where load is applied linear over time

    # setting up the problem

    experiment = AmMultipleLayers(setup_parameters)

    problem = ConcreteAMFC(experiment, setup_parameters, material_law, pv_name=file_name, pv_path=data_path)
    problem.set_initial_path(0.0)

    problem.add_sensor(ReactionForceSensor())
    problem.add_sensor(StressSensor([problem.p["layer_length"] / 2, 0, 0]))
    problem.add_sensor(DisplacementSensor([problem.p["layer_length"] / 2, 0, problem.p["layer_height"]]))

    E_o_time = []
    total_time = 60 * ureg("s")
    while problem.time <= total_time.to_base_units().magnitude:
        problem.solve()
        problem.pv_plot()
        # print("computed disp", problem.time, problem.fields.displacement.x.array[:].min())
        E_o_time.append(problem.modulus.vector.array[:].max())

    # check reaction force
    force_bottom_y = np.array(problem.sensors["ReactionForceSensor"].data)[:, -1]
    dead_load = (
        problem.p["g"]
        * problem.p["rho"]
        * problem.p["layer_length"]
        * problem.p["num_layers"]
        * problem.p["layer_height"]
        * problem.p["layer_width"]
    )

    # # dead load of full structure
    print("Check", force_bottom_y, dead_load)
    assert force_bottom_y[-1] == pytest.approx(-dead_load)

    # check stresses change
    sig_o_time = np.array(problem.sensors["StressSensor"].data)[:, 2]  # zz
    # print(sig_o_time)
    disp_o_time = np.array(problem.sensors["DisplacementSensor"].data)[:, 2]  # zz
    # print(disp_o_time)

    if factor == 1:
        # instance loading -> no changes
        assert sum(np.diff(sig_o_time)) == pytest.approx(0, abs=1e-8)
    elif factor > 1:
        # ratio sig/eps t=0 to sig/eps t=0+dt
        steps = len(np.where(abs(np.diff(sig_o_time)) > 1e-8)[0][:])
        assert steps == pytest.approx(factor - 1, abs=1e-8)
        # after loading steps nothing should change anymore
        assert sum(np.diff(sig_o_time)[factor - 1 : :]) == pytest.approx(0, abs=1e-8)

    if mat == "linear_elastic":
        # no changes in displacements after loading finshed
        assert sum(np.diff(disp_o_time)[factor - 1 : :]) == pytest.approx(0, abs=1e-8)
        # changing of material parameters
        if problem.p["time_fct"] == "linear":
            print(
                "check linear time dependency of Emodul",
                np.diff(E_o_time).mean(),
                (problem.p["A_E"] * problem.p["dt"]) / problem.p["E"],
            )
            assert np.isclose(np.diff(E_o_time).mean(), problem.p["A_E"] * problem.p["dt"] / problem.p["E"], rtol=1e-2)
    elif mat == "visco_Kelvin":
        # check for creep deformation over time
        # print("diff disp", np.diff(disp_o_time)[factor - 1 : :])
        assert sum(np.diff(disp_o_time)[factor - 1 : :]) != pytest.approx(0, abs=1e-8)
        assert abs(np.diff(disp_o_time)[factor - 1 : :][0]) > abs(np.diff(disp_o_time)[factor - 1 : :][-1])
        # changing of material parameters
        # print("E_o_time", E_o_time)
        if problem.p["time_fct"] == "linear":
            print(
                "check linear time dependency of Emodul",
                np.diff(E_o_time).mean(),
                (problem.p["A_E0"] * problem.p["dt"]),
            )
            assert np.isclose(np.diff(E_o_time).mean(), problem.p["A_E0"] * problem.p["dt"], rtol=1e-2)

    elif mat == "visco_Maxwell":
        # check for creep deformation over time
        # print("diff disp", np.diff(disp_o_time)[factor - 1 : :])
        assert sum(np.diff(disp_o_time)[factor - 1 : :]) != pytest.approx(0, abs=1e-8)
        assert abs(np.diff(disp_o_time)[factor - 1 : :][0]) > abs(np.diff(disp_o_time)[factor - 1 : :][-1])
        # changing of material parameters
        # print("E_o_time", E_o_time)
        if problem.p["time_fct"] == "linear":
            print(
                "check linear time dependency of Emodul",
                np.diff(E_o_time).mean(),
                (problem.p["A_E0"] * problem.p["dt"]),
            )
            assert np.isclose(np.diff(E_o_time).mean(), problem.p["A_E0"] * problem.p["dt"], rtol=1e-2)


@pytest.mark.parametrize("mat", ["linear_elastic", "visco_Kelvin", "visco_Maxwell", "mises"])
@pytest.mark.parametrize("factor", [1, 2])
def test_am_multiple_layer(
    mat: Literal["linear_elastic", "visco_Kelvin", "visco_Maxwell", "mises"], factor: int, plot: bool = False
) -> None:
    """multiple layer test

    several layers building over time one layer at once

    Args:
        mat: material law string in the moment only linear elastic
        factor: length of load_time = factor * dt

    """

    # setup paths and directories
    data_dir = "data_files"
    data_path = Path(__file__).parent / data_dir

    # define file name and path for paraview output
    file_name = "test_am_multiple_layer"
    files = [data_path / (file_name + ".xdmf"), data_path / (file_name + ".h5")]
    # delete file if it exists (only relevant for local tests)
    for file in files:
        if file.is_file():
            os.remove(file)

    # defining parameters
    setup_parameters, material_law = set_test_parameters(mat)

    # solving parameters
    time_layer = 20 * ureg("s")  # time to build one layer
    setup_parameters["dt"] = time_layer / 5
    assert factor < time_layer / setup_parameters["dt"], "factor is bigger then time for on layer"
    setup_parameters["load_time"] = factor * setup_parameters["dt"]  # interval where load is applied linear over time

    # setting up the problem
    experiment = AmMultipleLayers(setup_parameters)

    problem = ConcreteAMFC(experiment, setup_parameters, material_law, pv_name=file_name, pv_path=data_path)

    # initial path function describing layer activation
    path_activation = define_path(
        problem, time_layer.magnitude, t_0=-(setup_parameters["num_layers"].magnitude - 1) * time_layer.magnitude
    )
    problem.set_initial_path(path_activation)

    problem.add_sensor(ReactionForceSensor())
    problem.add_sensor(StressSensor([problem.p["layer_length"] / 2, 0, 0]))
    problem.add_sensor(DisplacementSensor([problem.p["layer_length"] / 2, 0, problem.p["layer_height"]]))

    total_time = setup_parameters["num_layers"] * time_layer
    while problem.time <= total_time.to_base_units().magnitude:
        problem.solve()
        problem.pv_plot()
        print("computed disp", problem.time, problem.fields.displacement.x.array[:].min())

    # check residual force bottom
    force_bottom_y = np.array(problem.sensors["ReactionForceSensor"].data)[:, -1]
    dead_load = (
        problem.p["g"]
        * problem.p["rho"]
        * problem.p["layer_length"]
        * problem.p["num_layers"]
        * problem.p["layer_height"]
        * problem.p["layer_width"]
    )

    print("check", force_bottom_y[-1], dead_load)
    # assert force_bottom_y[-1] == pytest.approx(-dead_load)

    # check E modulus evolution over structure (each layer different E)
    if mat.lower() == "linear_elastic":
        if problem.p["time_fct"] == "linear":
            E_bottom_layer = ConcreteAMFC.param_time_fkt(
                problem.time, {"P0": problem.p["E"], "A_P": problem.p["A_E"]}, "linear"
            )
            time_upper = problem.time - (problem.p["num_layers"] - 1) * time_layer.magnitude
            E_upper_layer = ConcreteAMFC.param_time_fkt(
                time_upper, {"P0": problem.p["E"], "A_P": problem.p["A_E"]}, "linear"
            )
            print("E_bottom, E_upper", E_bottom_layer, E_upper_layer)
            print(problem.modulus.vector.array[:].min(), problem.modulus.vector.array[:].max())
            assert problem.modulus.vector.array[:].min() * problem.p["E"] == pytest.approx(E_upper_layer)
            assert problem.modulus.vector.array[:].max() * problem.p["E"] == pytest.approx(E_bottom_layer)
        #
    if plot:
        # example plotting
        disp = np.array(problem.sensors["DisplacementSensor"].data)[:, -1]
        time = []
        [time.append(ti) for ti in problem.sensors["DisplacementSensor"].time]

        import matplotlib.pylab as plt

        plt.figure(1)
        plt.plot([0] + time, [0] + list(disp), "*-r")
        plt.xlabel("process time")
        plt.ylabel("displacement")
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
    if prob.p["dim"] == 2:
        v_cg.interpolate(lambda x: (x[0], x[1]))
    elif prob.p["dim"] == 3:
        v_cg.interpolate(lambda x: (x[0], x[1], x[2]))
    positions = QuadratureEvaluator(v_cg, prob.mesh, prob.rule)
    x = positions.evaluate()
    dof_map = np.reshape(x.flatten(), [len(q_path), prob.p["dim"]])

    # select layers
    if prob.p["dim"] == 2:
        # only by layer height - y
        h_CO = np.array(dof_map)[:, 1]
    elif prob.p["dim"] == 3:
        # only by layer height - z
        h_CO = np.array(dof_map)[:, 2]
    h_min = np.arange(0, prob.p["num_layers"] * prob.p["layer_height"], prob.p["layer_height"])
    h_max = np.arange(
        prob.p["layer_height"],
        (prob.p["num_layers"] + 1) * prob.p["layer_height"],
        prob.p["layer_height"],
    )
    # print("h_CO", h_CO)
    # print("h_min", h_min)
    # print("h_max", h_max)
    new_path = np.zeros_like(q_path)
    EPS = 1e-8
    for i in range(0, len(h_min)):
        layer_index = np.where((h_CO > h_min[i] - EPS) & (h_CO <= h_max[i] + EPS))
        new_path[layer_index] = t_0 + (prob.p["num_layers"] - 1 - i) * t_diff

    q_path = new_path

    return q_path


if __name__ == "__main__":
    # test_am_single_layer("linear_elastic", 2)
    # test_am_single_layer("visco_Kelvin", 2)
    # test_am_single_layer("visco_Maxwell", 2)
    # test_am_multiple_layer("linear_elastic", 1, plot=True)
    # test_am_multiple_layer("visco_Kelvin", 2, plot=False)
    # test_am_multiple_layer("visco_Maxwell", 2, plot=False)

    test_am_multiple_layer("mises", 1, plot=True)
